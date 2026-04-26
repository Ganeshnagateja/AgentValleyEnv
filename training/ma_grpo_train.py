"""Multi-agent GRPO-style policy optimization for MultiAgentValleyEnv.

This trainer keeps the clean ``multi_agent_grpo`` naming used in the OpenEnv
manifest, while borrowing the stronger candidate-selection loop from the
expanded prototype: each role samples a small group of candidate composite
actions, scores those candidates through real environment rollouts, updates one
role policy with group-relative advantages, then executes the best candidate in
the live multi-agent episode.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.distributions import Categorical, kl_divergence

from env.action_space import action_to_index, index_to_action
from env.agents import AGENT_IDS, AGENT_SPECS
from env.multi_agent_env import MultiAgentValleyEnv
from training.common import (
    ARTIFACT_ROOT,
    append_jsonl,
    mean,
    project_relative,
    reset_file,
    resolve_difficulty,
    score_components,
    utc_now,
)
from training.feature_encoder import encode_observation
from training.neural_policy import NeuralPolicy, save_checkpoint, select_action


MetricCallback = Callable[[dict[str, Any]], None]


@dataclass
class MAGRPOConfig:
    episodes: int = 20
    difficulty: str = "mixed"
    seed: int = 42
    group_size: int = 4
    learning_rate: float = 0.0015
    hidden_dim: int = 64
    artifact_dir: Path = ARTIFACT_ROOT / "multi_agent_grpo"
    clip_range: float = 0.2
    kl_coef: float = 0.02
    entropy_coef: float = 0.01
    optimization_epochs: int = 2
    reset_metrics: bool = True


def role_default_action(agent_id: str) -> dict[str, Any]:
    """Return a safe role-aligned fallback action for one specialist."""
    spec = AGENT_SPECS[agent_id]
    focus = str(spec.preferred_focus)
    primary = str(spec.preferred_action)
    cooperation = str(spec.preferred_cooperation)
    focus_phrase = "no resource" if focus == "none" else f"{focus} focus"
    return {
        "primary_action": primary,
        "focus_resource": focus,
        "cooperation_mode": cooperation,
        "risk_posture": "cautious" if agent_id == "warrior" else "balanced",
        "rationale": f"{agent_id} follows its role by choosing {primary} with {focus_phrase}.",
    }


def role_default_actions() -> dict[str, dict[str, Any]]:
    return {agent_id: role_default_action(agent_id) for agent_id in AGENT_IDS}


class MAGRPOTrainer:
    """One small policy per role, trained with group-relative candidate rewards."""

    def __init__(self, config: MAGRPOConfig):
        self.config = config
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.policies = {agent_id: NeuralPolicy(hidden_dim=config.hidden_dim) for agent_id in AGENT_IDS}
        self.reference_policies = {agent_id: NeuralPolicy(hidden_dim=config.hidden_dim) for agent_id in AGENT_IDS}
        self.optimizers = {
            agent_id: torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
            for agent_id, policy in self.policies.items()
        }
        for agent_id in AGENT_IDS:
            self.reference_policies[agent_id].load_state_dict(self.policies[agent_id].state_dict())
            self.reference_policies[agent_id].eval()
            for param in self.reference_policies[agent_id].parameters():
                param.requires_grad_(False)

        self.metrics_path = config.artifact_dir / "metrics.jsonl"
        self.checkpoint_paths = {
            agent_id: config.artifact_dir / f"agent_{agent_id}" / "policy.pt"
            for agent_id in AGENT_IDS
        }
        self.update_step = 0

    def _sample_candidates(
        self,
        agent_id: str,
        observation: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
        sample_count = max(0, self.config.group_size - 1)
        feature_count = max(sample_count, 1)
        features = torch.tensor(
            [encode_observation(observation) for _ in range(feature_count)],
            dtype=torch.float32,
        )
        logits = self.policies[agent_id](features)
        dist = Categorical(logits=logits)
        if sample_count > 0:
            sampled_indices = dist.sample()[:sample_count]
            sampled_log_probs = dist.log_prob(sampled_indices)[:sample_count].detach()
        else:
            sampled_indices = torch.empty(0, dtype=torch.long)
            sampled_log_probs = torch.empty(0)

        # Always include a role-aligned candidate. This gives the trainer a
        # stable anchor and makes before/after behavior easy to explain.
        default = role_default_action(agent_id)
        default_index = torch.tensor([action_to_index(default)], dtype=torch.long)
        with torch.no_grad():
            default_features = torch.tensor([encode_observation(observation)], dtype=torch.float32)
            default_dist = Categorical(logits=self.policies[agent_id](default_features))
            default_log_prob = default_dist.log_prob(default_index)

        action_indices = torch.cat([sampled_indices, default_index])
        old_log_probs = torch.cat([sampled_log_probs, default_log_prob.detach()])
        candidates = [index_to_action(int(index.item())).model_dump(mode="json") for index in action_indices]
        return candidates, action_indices, old_log_probs

    def _score_candidate(
        self,
        difficulty: str,
        episode: int,
        history: list[dict[str, dict[str, Any]]],
        agent_id: str,
        candidate: dict[str, Any],
    ) -> tuple[float, bool, dict[str, Any]]:
        env = MultiAgentValleyEnv(difficulty=difficulty, episode_index=episode - 1, seed=self.config.seed)
        env.reset(episode_index=episode - 1, seed=self.config.seed)
        info: dict[str, Any] = {}
        for joint_action in history:
            _obs, _rewards, dones, info = env.step(copy.deepcopy(joint_action))
            if bool(dones.get("__all__")):
                return 0.0, True, info

        scored_actions = role_default_actions()
        scored_actions[agent_id] = candidate
        _next_obs, rewards, dones, info = env.step(scored_actions)
        return float(rewards[agent_id]), bool(dones.get("__all__")), info

    def _update_from_group(
        self,
        agent_id: str,
        observation: dict[str, Any],
        action_indices: torch.Tensor,
        rewards: list[float],
        old_log_probs: torch.Tensor,
    ) -> dict[str, float]:
        features = torch.tensor([encode_observation(observation) for _ in rewards], dtype=torch.float32)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        reward_std = reward_tensor.std(unbiased=False)
        advantages = reward_tensor - reward_tensor.mean()
        if float(reward_std.item()) > 1e-8:
            advantages = advantages / (reward_std + 1e-8)

        policy = self.policies[agent_id]
        reference = self.reference_policies[agent_id]
        optimizer = self.optimizers[agent_id]
        policy_loss = torch.tensor(0.0)
        kl = torch.tensor(0.0)
        entropy = torch.tensor(0.0)

        for _ in range(self.config.optimization_epochs):
            logits = policy(features)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(action_indices)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
            policy_loss = -torch.mean(torch.minimum(ratio * advantages, clipped_ratio * advantages))

            with torch.no_grad():
                ref_dist = Categorical(logits=reference(features))
            kl = kl_divergence(dist, ref_dist).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + self.config.kl_coef * kl - self.config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            self.update_step += 1

        return {
            "policy_loss": float(policy_loss.detach().item()),
            "kl_divergence": float(kl.detach().item()),
            "entropy": float(entropy.detach().item()),
            "mean_group_reward": float(reward_tensor.mean().item()),
            "reward_std": float(reward_std.item()),
        }

    def train_episode(self, difficulty: str, episode: int) -> dict[str, Any]:
        env = MultiAgentValleyEnv(difficulty=difficulty, episode_index=episode - 1, seed=self.config.seed)
        observations = env.reset(episode_index=episode - 1, seed=self.config.seed)
        done = False
        info: dict[str, Any] = {}
        steps = 0
        history: list[dict[str, dict[str, Any]]] = []
        total_team_reward = 0.0
        cooperation_steps = 0
        per_agent_rewards = {agent_id: 0.0 for agent_id in AGENT_IDS}
        per_agent_role_rewards = {agent_id: [] for agent_id in AGENT_IDS}
        per_agent_losses = {agent_id: [] for agent_id in AGENT_IDS}
        per_agent_kls = {agent_id: [] for agent_id in AGENT_IDS}
        per_agent_entropies = {agent_id: [] for agent_id in AGENT_IDS}
        group_means: list[float] = []
        group_stds: list[float] = []

        while not done:
            joint_action: dict[str, dict[str, Any]] = {}
            for agent_id in AGENT_IDS:
                candidates, action_indices, old_log_probs = self._sample_candidates(agent_id, observations[agent_id])
                candidate_rewards = [
                    self._score_candidate(difficulty, episode, history, agent_id, candidate)[0]
                    for candidate in candidates
                ]
                update_stats = self._update_from_group(
                    agent_id,
                    observations[agent_id],
                    action_indices,
                    candidate_rewards,
                    old_log_probs,
                )
                per_agent_losses[agent_id].append(update_stats["policy_loss"])
                per_agent_kls[agent_id].append(update_stats["kl_divergence"])
                per_agent_entropies[agent_id].append(update_stats["entropy"])
                group_means.append(update_stats["mean_group_reward"])
                group_stds.append(update_stats["reward_std"])

                best_index = max(range(len(candidates)), key=lambda idx: candidate_rewards[idx])
                joint_action[agent_id] = candidates[best_index]

            next_observations, rewards, dones, info = env.step(joint_action)
            history.append(copy.deepcopy(joint_action))
            for agent_id, reward in rewards.items():
                per_agent_rewards[agent_id] += float(reward)
            total_team_reward += sum(float(value) for value in rewards.values())
            cooperation_bonus = float(info.get("cooperation_bonus", 0.0))
            if cooperation_bonus > 0:
                cooperation_steps += 1
            role_rewards = info.get("individual_role_rewards") or {}
            for agent_id in AGENT_IDS:
                per_agent_role_rewards[agent_id].append(float(role_rewards.get(agent_id, 0.0)))

            observations = next_observations
            done = bool(dones.get("__all__"))
            steps += 1

        episode_result = info.get("episode_result") or {}
        score_bits = score_components(episode_result)
        updated_at = utc_now()
        self.save_checkpoints(episode=episode, updated_at=updated_at)

        metric: dict[str, Any] = {
            "mode": "multi_agent_grpo",
            "episode": episode,
            "update_step": self.update_step,
            "difficulty": difficulty,
            "total_reward": round(total_team_reward, 4),
            "total_team_reward": round(total_team_reward, 4),
            "cooperation_rate": round(cooperation_steps / max(steps, 1), 6),
            "mean_group_reward": round(mean(group_means), 6),
            "reward_std": round(mean(group_stds), 6),
            "policy_loss": round(mean(value for values in per_agent_losses.values() for value in values), 6),
            "kl_divergence": round(mean(value for values in per_agent_kls.values() for value in values), 6),
            "entropy": round(mean(value for values in per_agent_entropies.values() for value in values), 6),
            "task_score": float(episode_result.get("task_score", 0.0)),
            "final_status": episode_result.get("final_status", "unknown"),
            "steps": steps,
            "action_accuracy": score_bits["action_accuracy"],
            "safety_score": score_bits["safety_score"],
            "checkpoint_path": project_relative(self.config.artifact_dir),
            "metrics_path": project_relative(self.metrics_path),
            "updated_at": updated_at,
        }
        agent_metrics: dict[str, dict[str, Any]] = {}
        for agent_id in AGENT_IDS:
            agent_metrics[agent_id] = {
                "total_reward": round(per_agent_rewards[agent_id], 4),
                "avg_reward": round(per_agent_rewards[agent_id] / max(steps, 1), 6),
                "role_reward": round(mean(per_agent_role_rewards[agent_id]), 6),
                "policy_loss": round(mean(per_agent_losses[agent_id]), 6),
                "kl_divergence": round(mean(per_agent_kls[agent_id]), 6),
                "entropy": round(mean(per_agent_entropies[agent_id]), 6),
                "checkpoint_path": project_relative(self.checkpoint_paths[agent_id]),
            }
            metric[f"{agent_id}_total_reward"] = agent_metrics[agent_id]["total_reward"]
            metric[f"{agent_id}_role_reward"] = agent_metrics[agent_id]["role_reward"]
            metric[f"{agent_id}_policy_loss"] = agent_metrics[agent_id]["policy_loss"]
            metric[f"{agent_id}_checkpoint_path"] = agent_metrics[agent_id]["checkpoint_path"]
        metric["agent_metrics"] = agent_metrics
        return metric

    def save_checkpoints(self, episode: int, updated_at: str) -> None:
        for agent_id in AGENT_IDS:
            save_checkpoint(
                self.checkpoint_paths[agent_id],
                self.policies[agent_id],
                self.optimizers[agent_id],
                metadata={"mode": "multi_agent_grpo", "agent_id": agent_id, "episode": episode, "updated_at": updated_at},
            )

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

    def trained_action(self, agent_id: str, observation: dict[str, Any]) -> dict[str, Any]:
        _index, action = select_action(self.policies[agent_id], observation, deterministic=True)
        return action


def train_ma_grpo(**kwargs: Any) -> list[dict[str, Any]]:
    return MAGRPOTrainer(MAGRPOConfig(**kwargs)).train()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train four multi-agent GRPO-style policies on MultiAgentValleyEnv")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", default="mixed", choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    args = parser.parse_args()

    trainer = MAGRPOTrainer(
        MAGRPOConfig(
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
