"""Train the lightweight neural policy with real environment rewards."""

from __future__ import annotations

import argparse
import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.distributions import Categorical

from env.action_space import index_to_action
from env.environment import AgentValleyEnv
from training.common import ARTIFACT_ROOT, append_jsonl, project_relative, reset_file, resolve_difficulty, score_components, utc_now
from training.feature_encoder import encode_observation
from training.neural_policy import NeuralPolicy, save_checkpoint


MetricCallback = Callable[[dict[str, Any]], None]


@dataclass
class NeuralPolicyConfig:
    episodes: int = 20
    difficulty: str = "mixed"
    seed: int = 42
    learning_rate: float = 0.002
    gamma: float = 0.95
    entropy_coef: float = 0.01
    hidden_dim: int = 64
    artifact_dir: Path = ARTIFACT_ROOT / "neural_policy"
    reset_metrics: bool = True


class NeuralPolicyTrainer:
    def __init__(self, config: NeuralPolicyConfig):
        self.config = config
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(False)
        self.policy = NeuralPolicy(hidden_dim=config.hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.checkpoint_path = config.artifact_dir / "policy.pt"
        self.metrics_path = config.artifact_dir / "metrics.jsonl"
        self.update_step = 0

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns: list[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        tensor = torch.tensor(returns, dtype=torch.float32)
        if tensor.numel() > 1:
            std = tensor.std(unbiased=False)
            if float(std.item()) > 1e-8:
                tensor = (tensor - tensor.mean()) / (std + 1e-8)
            else:
                tensor = tensor - tensor.mean()
        return tensor

    def train_episode(self, difficulty: str, episode: int) -> dict[str, Any]:
        env = AgentValleyEnv(difficulty=difficulty, episode_index=episode - 1, seed=self.config.seed)
        observation = env.reset(episode_index=episode - 1, seed=self.config.seed)
        done = False
        info: dict[str, Any] = {}
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []
        total_reward = 0.0
        steps = 0

        while not done:
            features = torch.tensor(encode_observation(observation), dtype=torch.float32).unsqueeze(0)
            logits = self.policy(features)
            dist = Categorical(logits=logits)
            action_index = dist.sample()
            action = index_to_action(int(action_index.item())).model_dump(mode="json")
            next_observation, reward, done, info = env.step(action)
            log_probs.append(dist.log_prob(action_index).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            rewards.append(float(reward))
            total_reward += float(reward)
            steps += 1
            observation = next_observation

        returns = self._discounted_returns(rewards)
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        policy_loss = -(log_prob_tensor * returns).mean()
        entropy = entropy_tensor.mean()
        loss = policy_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.update_step += 1

        episode_result = info.get("episode_result") or {}
        score_bits = score_components(episode_result)
        metric = {
            "mode": "neural_policy",
            "episode": episode,
            "update_step": self.update_step,
            "difficulty": difficulty,
            "total_reward": round(total_reward, 4),
            "policy_loss": round(float(policy_loss.detach().item()), 6),
            "entropy": round(float(entropy.detach().item()), 6),
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
            metadata={"mode": "neural_policy", "episode": episode, "updated_at": metric["updated_at"]},
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


def train_neural_policy(**kwargs: Any) -> list[dict[str, Any]]:
    return NeuralPolicyTrainer(NeuralPolicyConfig(**kwargs)).train()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a neural policy on AgentValleyEnv")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", default="mixed", choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    args = parser.parse_args()

    trainer = NeuralPolicyTrainer(
        NeuralPolicyConfig(
            episodes=args.episodes,
            difficulty=args.difficulty,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
    )
    metrics = trainer.train()
    print(json.dumps({"episodes": len(metrics), "latest": metrics[-1] if metrics else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
