"""Small inference helper for AgentValleyEnv demos."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from baseline_eval import RuleBasedAgent
from env.environment import AgentValleyEnv


def predict_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deterministic baseline action for a single observation."""
    return RuleBasedAgent().act(observation).model_dump(mode="json")


def run_demo(difficulty: str = "easy", episode_index: int = 0, seed: int = 42) -> Dict[str, Any]:
    env = AgentValleyEnv(difficulty=difficulty, episode_index=episode_index, seed=seed)
    agent = RuleBasedAgent()
    obs = env.reset()
    done = False
    trace = []
    info: Dict[str, Any] = {}
    while not done:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        trace.append(
            {
                "observation": obs,
                "action": action.model_dump(mode="json"),
                "reward": reward,
                "done": done,
                "reward_breakdown": info.get("reward_breakdown", {}),
            }
        )
        obs = next_obs
    return {"trace": trace, "episode_result": info.get("episode_result", {})}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an AgentValleyEnv inference demo")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.task, args.episode, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
