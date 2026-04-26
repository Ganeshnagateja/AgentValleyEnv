"""Load trained backend policies and evaluate them in AgentValleyEnv."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import torch

from env.action_space import action_count, index_to_action, random_action
from env.environment import AgentValleyEnv
from training.common import ARTIFACT_ROOT, project_relative, score_components, utc_now
from training.feature_encoder import discretize_observation
from training.neural_policy import load_checkpoint, select_action


CHECKPOINTS = {
    "q_learning": ARTIFACT_ROOT / "q_learning" / "q_table.json",
    "neural_policy": ARTIFACT_ROOT / "neural_policy" / "policy.pt",
    "grpo": ARTIFACT_ROOT / "grpo" / "policy.pt",
}


def _q_action(observation: Mapping[str, Any]) -> tuple[int, dict[str, Any], bool]:
    path = CHECKPOINTS["q_learning"]
    state_key = discretize_observation(observation)
    if not path.exists():
        return 0, index_to_action(0).model_dump(mode="json"), False
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("q_table", {}).get(state_key)
    if not values:
        return 0, index_to_action(0).model_dump(mode="json"), True
    best_value = max(values)
    action_index = values.index(best_value)
    return int(action_index), index_to_action(action_index).model_dump(mode="json"), True


def policy_action(
    mode: str,
    observation: Mapping[str, Any],
    deterministic: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    if mode == "q_learning":
        action_index, action, checkpoint_found = _q_action(observation)
        checkpoint_path = CHECKPOINTS[mode]
    elif mode in {"neural_policy", "grpo"}:
        checkpoint_path = CHECKPOINTS[mode]
        torch.manual_seed(seed)
        policy, metadata = load_checkpoint(checkpoint_path)
        action_index, action = select_action(policy, observation, deterministic=deterministic)
        checkpoint_found = bool(metadata.get("checkpoint_found"))
    else:
        raise ValueError(f"Unknown policy mode '{mode}'")

    return {
        "mode": mode,
        "action_index": action_index,
        "action": action,
        "action_count": action_count(),
        "checkpoint_path": project_relative(checkpoint_path),
        "checkpoint_found": checkpoint_found,
        "updated_at": utc_now(),
    }


def evaluate_policy(
    mode: str,
    difficulty: str = "easy",
    episodes: int = 3,
    seed: int = 42,
    deterministic: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    for episode in range(1, episodes + 1):
        env = AgentValleyEnv(difficulty=difficulty, episode_index=episode - 1, seed=seed)
        observation = env.reset(episode_index=episode - 1, seed=seed)
        done = False
        info: dict[str, Any] = {}
        total_reward = 0.0
        steps = 0
        while not done:
            if mode == "random":
                action = random_action(rng).model_dump(mode="json")
            else:
                action = policy_action(mode, observation, deterministic=deterministic, seed=seed + episode)["action"]
            observation, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
        episode_result = info.get("episode_result") or {}
        bits = score_components(episode_result)
        results.append(
            {
                "episode": episode,
                "difficulty": difficulty,
                "total_reward": round(total_reward, 4),
                "task_score": float(episode_result.get("task_score", 0.0)),
                "final_status": episode_result.get("final_status", "unknown"),
                "steps": steps,
                "action_accuracy": bits["action_accuracy"],
                "safety_score": bits["safety_score"],
            }
        )

    avg_reward = sum(item["total_reward"] for item in results) / max(len(results), 1)
    avg_score = sum(item["task_score"] for item in results) / max(len(results), 1)
    return {
        "mode": mode,
        "difficulty": difficulty,
        "episodes": episodes,
        "average_total_reward": round(avg_reward, 4),
        "average_task_score": round(avg_score, 4),
        "results": results,
        "updated_at": utc_now(),
    }
