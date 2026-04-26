"""Real tabular Q-learning for AgentValleyEnv."""

from __future__ import annotations

import argparse
import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from env.action_space import action_count, index_to_action
from env.environment import AgentValleyEnv
from training.common import ARTIFACT_ROOT, append_jsonl, ensure_parent, project_relative, reset_file, resolve_difficulty, score_components, utc_now
from training.feature_encoder import discretize_observation


MetricCallback = Callable[[dict[str, Any]], None]


@dataclass
class QLearningConfig:
    episodes: int = 20
    difficulty: str = "mixed"
    seed: int = 42
    alpha: float = 0.25
    gamma: float = 0.95
    epsilon: float = 0.35
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.97
    artifact_dir: Path = ARTIFACT_ROOT / "q_learning"
    reset_metrics: bool = True


class QLearningTrainer:
    def __init__(self, config: QLearningConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.action_dim = action_count()
        self.q_table: dict[str, list[float]] = {}
        self.q_table_path = config.artifact_dir / "q_table.json"
        self.metrics_path = config.artifact_dir / "metrics.jsonl"

    def load(self) -> None:
        if self.q_table_path.exists():
            payload = json.loads(self.q_table_path.read_text(encoding="utf-8"))
            self.q_table = {key: [float(v) for v in values] for key, values in payload.get("q_table", {}).items()}

    def save(self) -> None:
        ensure_parent(self.q_table_path)
        payload = {
            "action_dim": self.action_dim,
            "updated_at": utc_now(),
            "q_table": self.q_table,
        }
        self.q_table_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def ensure_state(self, state_key: str) -> list[float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0 for _ in range(self.action_dim)]
        return self.q_table[state_key]

    def select_action(self, state_key: str, epsilon: float) -> int:
        values = self.ensure_state(state_key)
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.action_dim)
        best_value = max(values)
        return values.index(best_value)

    def bellman_update(self, state_key: str, action_index: int, reward: float, next_state_key: str, done: bool) -> float:
        values = self.ensure_state(state_key)
        old_value = values[action_index]
        next_values = self.ensure_state(next_state_key)
        target = reward if done else reward + self.config.gamma * max(next_values)
        new_value = old_value + self.config.alpha * (target - old_value)
        values[action_index] = round(new_value, 8)
        return values[action_index]

    def q_stats(self) -> tuple[float, float]:
        all_values = [value for values in self.q_table.values() for value in values]
        if not all_values:
            return 0.0, 0.0
        return sum(all_values) / len(all_values), max(all_values)

    def train(
        self,
        stop_event: threading.Event | None = None,
        metric_callback: MetricCallback | None = None,
    ) -> list[dict[str, Any]]:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.config.reset_metrics:
            reset_file(self.metrics_path)
            self.q_table = {}
        else:
            self.load()

        metrics: list[dict[str, Any]] = []
        epsilon = self.config.epsilon

        for episode in range(1, self.config.episodes + 1):
            if stop_event and stop_event.is_set():
                break

            difficulty = resolve_difficulty(self.config.difficulty, episode)
            env = AgentValleyEnv(difficulty=difficulty, episode_index=episode - 1, seed=self.config.seed)
            observation = env.reset(episode_index=episode - 1, seed=self.config.seed)
            done = False
            info: dict[str, Any] = {}
            total_reward = 0.0
            steps = 0

            while not done:
                state_key = discretize_observation(observation)
                action_index = self.select_action(state_key, epsilon)
                action = index_to_action(action_index).model_dump(mode="json")
                next_observation, reward, done, info = env.step(action)
                next_state_key = discretize_observation(next_observation)
                self.bellman_update(state_key, action_index, float(reward), next_state_key, done)
                total_reward += float(reward)
                steps += 1
                observation = next_observation
                if stop_event and stop_event.is_set():
                    break

            episode_result = info.get("episode_result") or {}
            mean_q, max_q = self.q_stats()
            score_bits = score_components(episode_result)
            metric = {
                "mode": "q_learning",
                "episode": episode,
                "difficulty": difficulty,
                "total_reward": round(total_reward, 4),
                "task_score": float(episode_result.get("task_score", 0.0)),
                "final_status": episode_result.get("final_status", "stopped" if stop_event and stop_event.is_set() else "unknown"),
                "steps": steps,
                "epsilon": round(epsilon, 6),
                "mean_q_value": round(mean_q, 6),
                "max_q_value": round(max_q, 6),
                "action_accuracy": score_bits["action_accuracy"],
                "safety_score": score_bits["safety_score"],
                "checkpoint_path": project_relative(self.q_table_path),
                "metrics_path": project_relative(self.metrics_path),
                "updated_at": utc_now(),
            }
            append_jsonl(self.metrics_path, metric)
            self.save()
            metrics.append(metric)
            if metric_callback:
                metric_callback(metric)

            epsilon = max(self.config.min_epsilon, epsilon * self.config.epsilon_decay)

        return metrics


def train_q_learning(**kwargs: Any) -> list[dict[str, Any]]:
    config = QLearningConfig(**kwargs)
    return QLearningTrainer(config).train()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning policy on AgentValleyEnv")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", default="mixed", choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.35)
    args = parser.parse_args()

    trainer = QLearningTrainer(
        QLearningConfig(
            episodes=args.episodes,
            difficulty=args.difficulty,
            seed=args.seed,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
        )
    )
    metrics = trainer.train()
    print(json.dumps({"episodes": len(metrics), "latest": metrics[-1] if metrics else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
