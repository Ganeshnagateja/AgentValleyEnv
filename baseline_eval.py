"""Reproducible baseline evaluator for AgentValleyEnv.

Usage:
  python baseline_eval.py --no-llm
  python baseline_eval.py --task hard --episodes 3 --seed 42
  python baseline_eval.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from env.environment import AgentValleyEnv
from env.schemas import Action


class RuleBasedAgent:
    """Deterministic no-key baseline that acts only on visible observations."""

    def act(self, obs: Dict[str, Any]) -> Action:
        action = "gather"
        focus = "food"
        cooperation = "share"
        risk = "balanced"

        combined_threat = (obs["threat_level"] + obs["event_severity"] + obs["region_danger"]) / 3.0

        if obs["average_energy"] < 0.28 and obs["threat_level"] < 0.65:
            action, focus, cooperation, risk = "rest", "none", "solo", "cautious"
        elif obs["threat_level"] >= 0.70:
            action, focus, cooperation, risk = "defend", "none", "protect", "cautious"
        elif obs["cooperation_index"] < 0.36 and (obs["event_severity"] > 0.45 or obs["threat_level"] > 0.35):
            action, focus, cooperation, risk = "cooperate", "none", "coordinate", "cautious"
        elif obs["gold_supply"] < 0.20 and obs["average_energy"] > 0.75 and obs["threat_level"] < 0.35 and obs["cooperation_index"] < 0.35:
            action, focus, cooperation, risk = "compete", "gold", "solo", "aggressive"
        elif obs["food_supply"] < 0.35:
            action, focus, cooperation, risk = "gather", "food", "share", "balanced" if combined_threat < 0.45 else "cautious"
        elif obs["wood_supply"] < 0.35:
            action, focus, cooperation, risk = "gather", "wood", "coordinate", "balanced"
        elif obs["stone_supply"] < 0.35:
            action, focus, cooperation, risk = "gather", "stone", "coordinate", "balanced"
        elif obs["ore_supply"] < 0.25:
            action, focus, cooperation, risk = "gather", "ore", "coordinate", "balanced"
        elif obs["market_volatility"] > 0.60 or obs["gold_supply"] < 0.25:
            action, focus, cooperation, risk = "trade", "gold", "share", "balanced"
        elif obs["defense_readiness"] < 0.50 and obs["wood_supply"] > 0.55 and obs["stone_supply"] > 0.55:
            action, focus, cooperation, risk = "build", "stone", "coordinate", "cautious" if combined_threat > 0.45 else "balanced"
        elif obs["average_energy"] > 0.78 and obs["threat_level"] < 0.40 and obs["region"] in {"wilderness", "mine"}:
            action, focus, cooperation, risk = "explore", "ore", "coordinate", "balanced"
        elif obs["event_severity"] > 0.55:
            action, focus, cooperation, risk = "cooperate", "none", "coordinate", "cautious"

        return Action(
            primary_action=action,
            focus_resource=focus,
            cooperation_mode=cooperation,
            risk_posture=risk,
            rationale=f"{action} selected from visible valley signals",
        )


class LLMAgent:
    """Optional OpenAI-backed agent with deterministic baseline fallback."""

    SYSTEM_PROMPT = (
        "You control a multi-agent valley RL environment. "
        "Choose the best JSON action with keys primary_action, focus_resource, "
        "cooperation_mode, risk_posture, rationale. "
        "Valid primary_action values: gather, trade, build, explore, rest, cooperate, compete, defend. "
        "Valid focus_resource values: none, food, wood, stone, gold, ore. "
        "Valid cooperation_mode values: solo, share, coordinate, protect. "
        "Valid risk_posture values: cautious, balanced, aggressive."
    )

    def __init__(self, model: str = "gpt-4o-mini"):
        self._fallback = RuleBasedAgent()
        self._ok = False
        self._model = model
        try:
            from openai import OpenAI

            self._client = OpenAI()
            self._ok = bool(os.getenv("OPENAI_API_KEY"))
        except Exception:
            self._ok = False

    def act(self, obs: Dict[str, Any]) -> Action:
        if not self._ok:
            return self._fallback.act(obs)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(obs, indent=2)},
                ],
                temperature=0.0,
                max_tokens=220,
                response_format={"type": "json_object"},
            )
            return Action(**json.loads(response.choices[0].message.content))
        except Exception:
            return self._fallback.act(obs)


def run_episode(agent: Any, difficulty: str, episode_index: int, seed: int) -> Dict[str, Any]:
    env = AgentValleyEnv(difficulty=difficulty, episode_index=episode_index, seed=seed)
    obs = env.reset()
    done = False
    info: Dict[str, Any] = {}

    while not done:
        action = agent.act(obs)
        obs, _reward, done, info = env.step(action)

    return info.get("episode_result", {})


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentValleyEnv baseline evaluator")
    parser.add_argument("--task", default="all", choices=["all", "easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    agent = RuleBasedAgent() if args.no_llm else LLMAgent(args.model)

    print()
    print("=" * 64)
    print("  AgentValleyEnv - Baseline Evaluation")
    print("=" * 64)

    all_scores: Dict[str, List[float]] = {}
    statuses: Dict[str, List[str]] = {}
    for difficulty in tasks:
        scores: List[float] = []
        statuses[difficulty] = []
        for episode in range(args.episodes):
            result = run_episode(agent, difficulty, episode, args.seed)
            score = float(result.get("task_score", 0.0))
            status = result.get("final_status", "unknown")
            scores.append(score)
            statuses[difficulty].append(status)
            print(f"  {difficulty:6s} ep={episode} score={score:.4f} status={status}")
        all_scores[difficulty] = scores

    print("-" * 64)
    overall: List[float] = []
    for difficulty, scores in all_scores.items():
        avg = sum(scores) / max(len(scores), 1)
        overall.extend(scores)
        print(f"  {difficulty.title():20s}: {avg:.4f}")
    average = sum(overall) / max(len(overall), 1)
    print(f"  {'Average':20s}: {average:.4f}")
    print("=" * 64)
    print()

    summary = {
        "tasks": {key: round(sum(value) / max(len(value), 1), 4) for key, value in all_scores.items()},
        "average": round(average, 4),
        "statuses": statuses,
        "final_output_labels": ["valley_stabilized", "valley_survived", "valley_collapsed"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
