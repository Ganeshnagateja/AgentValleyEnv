"""CPU-safe GRPO-style policy optimization for AgentValleyEnv.

This is not a transformer GRPOTrainer wrapper.  It is a small PyTorch policy
optimizer that keeps the key GRPO/PPO mechanics honest: sampled action groups,
real environment rewards, group-relative advantages, clipped log-probability
ratios, optional reference-policy KL, entropy bonus, and optimizer.step().
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.distributions import Categorical, kl_divergence

from env.action_space import index_to_action
from env.environment import AgentValleyEnv
from training.common import ARTIFACT_ROOT, append_jsonl, project_relative, reset_file, resolve_difficulty, score_components, utc_now
from training.feature_encoder import encode_observation
from training.neural_policy import NeuralPolicy, save_checkpoint


MetricCallback = Callable[[dict[str, Any]], None]


@dataclass
class GRPOConfig:
    episodes: int = 20
    difficulty: str = "mixed"
    seed: int = 42
    learning_rate: float = 0.0015
    group_size: int = 4
    clip_range: float = 0.2
    kl_coef: float = 0.02
    entropy_coef: float = 0.01
    optimization_epochs: int = 2
    hidden_dim: int = 64
    artifact_dir: Path = ARTIFACT_ROOT / "grpo"
    reset_metrics: bool = True


class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.config = config
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.policy = NeuralPolicy(hidden_dim=config.hidden_dim)
        self.reference_policy = NeuralPolicy(hidden_dim=config.hidden_dim)
        self.reference_policy.load_state_dict(self.policy.state_dict())
        self.reference_policy.eval()
        for param in self.reference_policy.parameters():
            param.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.checkpoint_path = config.artifact_dir / "policy.pt"
        self.metrics_path = config.artifact_dir / "metrics.jsonl"
        self.update_step = 0

    def _score_candidate(
        self,
        difficulty: str,
        episode: int,
        history: list[dict[str, Any]],
        candidate: dict[str, Any],
    ) -> tuple[float, bool, dict[str, Any]]:
        """Replay history into a fresh env and step one candidate action."""
        env = AgentValleyEnv(difficulty=difficulty, episode_index=episode - 1, seed=self.config.seed)
        env.reset(episode_index=episode - 1, seed=self.config.seed)
        done = False
        info: dict[str, Any] = {}
        for action in history:
            _obs, _reward, done, info = env.step(action)
            if done:
                return 0.0, done, info
        _next_obs, reward, done, info = env.step(candidate)
        return float(reward), done, info

    def _update_from_group(self, observation: dict[str, Any], action_indices: torch.Tensor, rewards: list[float], old_log_probs: torch.Tensor) -> dict[str, float]:
        features = torch.tensor([encode_observation(observation) for _ in rewards], dtype=torch.float32)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        reward_std = reward_tensor.std(unbiased=False)
        advantages = reward_tensor - reward_tensor.mean()
        if float(reward_std.item()) > 1e-8:
            advantages = advantages / (reward_std + 1e-8)

        policy_loss = torch.tensor(0.0)
        kl = torch.tensor(0.0)
        entropy = torch.tensor(0.0)
        for _ in range(self.config.optimization_epochs):
            logits = self.policy(features)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(action_indices)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
            policy_loss = -torch.mean(torch.minimum(ratio * advantages, clipped_ratio * advantages))

            with torch.no_grad():
                ref_logits = self.reference_policy(features)
                ref_dist = Categorical(logits=ref_logits)
            kl = kl_divergence(dist, ref_dist).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + self.config.kl_coef * kl - self.config.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.update_step += 1

        return {
            "policy_loss": float(policy_loss.detach().item()),
            "kl_divergence": float(kl.detach().item()),
            "entropy": float(entropy.detach().item()),
            "mean_group_reward": float(reward_tensor.mean().item()),
            "reward_std": float(reward_std.item()),
        }

    def train_episode(self, difficulty: str, episode: int) -> dict[str, Any]:
        env = AgentValleyEnv(difficulty=difficulty, episode_index=episode - 1, seed=self.config.seed)
        observation = env.reset(episode_index=episode - 1, seed=self.config.seed)
        done = False
        info: dict[str, Any] = {}
        total_reward = 0.0
        steps = 0
        history: list[dict[str, Any]] = []
        losses: list[float] = []
        kls: list[float] = []
        entropies: list[float] = []
        group_means: list[float] = []
        group_stds: list[float] = []

        while not done:
            features = torch.tensor([encode_observation(observation) for _ in range(self.config.group_size)], dtype=torch.float32)
            logits = self.policy(features)
            dist = Categorical(logits=logits)
            action_indices = dist.sample()
            old_log_probs = dist.log_prob(action_indices).detach()

            candidates = [index_to_action(int(index.item())).model_dump(mode="json") for index in action_indices]
            candidate_rewards: list[float] = []
            for candidate in candidates:
                reward, _candidate_done, _candidate_info = self._score_candidate(difficulty, episode, history, candidate)
                candidate_rewards.append(reward)

            update_stats = self._update_from_group(observation, action_indices, candidate_rewards, old_log_probs)
            losses.append(update_stats["policy_loss"])
            kls.append(update_stats["kl_divergence"])
            entropies.append(update_stats["entropy"])
            group_means.append(update_stats["mean_group_reward"])
            group_stds.append(update_stats["reward_std"])

            best_index = max(range(len(candidates)), key=lambda idx: candidate_rewards[idx])
            chosen_action = candidates[best_index]
            next_observation, reward, done, info = env.step(chosen_action)
            history.append(chosen_action)
            total_reward += float(reward)
            steps += 1
            observation = next_observation

        episode_result = info.get("episode_result") or {}
        score_bits = score_components(episode_result)
        metric = {
            "mode": "grpo",
            "episode": episode,
            "update_step": self.update_step,
            "difficulty": difficulty,
            "total_reward": round(total_reward, 4),
            "mean_group_reward": round(sum(group_means) / max(len(group_means), 1), 6),
            "reward_std": round(sum(group_stds) / max(len(group_stds), 1), 6),
            "policy_loss": round(sum(losses) / max(len(losses), 1), 6),
            "kl_divergence": round(sum(kls) / max(len(kls), 1), 6),
            "entropy": round(sum(entropies) / max(len(entropies), 1), 6),
            "task_score": float(episode_result.get("task_score", 0.0)),
            "final_status": episode_result.get("final_status", "unknown"),
            "steps": steps,
            "action_accuracy": score_bits["action_accuracy"],
            "safety_score": score_bits["safety_score"],
            "checkpoint_path": project_relative(self.checkpoint_path),
            "metrics_path": project_relative(self.metrics_path),
            "updated_at": utc_now(),
        }
        save_checkpoint(
            self.checkpoint_path,
            self.policy,
            self.optimizer,
            metadata={"mode": "grpo", "episode": episode, "updated_at": metric["updated_at"]},
        )
        return metric

    def train(
        self,
        stop_event: threading.Event | None = None,
        metric_callback: MetricCallback | None = None,
    ) -> list[dict[str, Any]]:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.config.reset_metrics:
            reset_file(self.metrics_path)

        metrics: list[dict[str, Any]] = []
        for episode in range(1, self.config.episodes + 1):
            if stop_event and stop_event.is_set():
                break
            difficulty = resolve_difficulty(self.config.difficulty, episode)
            metric = self.train_episode(difficulty, episode)
            append_jsonl(self.metrics_path, metric)
            metrics.append(metric)
            if metric_callback:
                metric_callback(metric)
        return metrics


def train_grpo(**kwargs: Any) -> list[dict[str, Any]]:
    return GRPOTrainer(GRPOConfig(**kwargs)).train()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a GRPO-style policy on AgentValleyEnv")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", default="mixed", choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    args = parser.parse_args()

    trainer = GRPOTrainer(
        GRPOConfig(
            episodes=args.episodes,
            difficulty=args.difficulty,
            seed=args.seed,
            group_size=args.group_size,
            learning_rate=args.learning_rate,
        )
    )
    metrics = trainer.train()
    print(json.dumps({"episodes": len(metrics), "latest": metrics[-1] if metrics else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
