"""DEPRECATED: TRL/Unsloth integration bridge for AgentValleyEnv.

Backend training does not use this file. The real CPU-safe GRPO-style trainer
lives in training/grpo_train.py. This module is retained only as an optional
reward-callback sketch for future transformer-scale experiments.

This file is intentionally lightweight so the project remains runnable on CPU
for reviewers, while still showing exactly how AgentValleyEnv connects to a
GRPO-style reward pipeline.

Common usage:
  python training/grpo_train_stub.py --dry-run

Real training outline:
  1. Install GPU stack: torch, transformers, trl, unsloth.
  2. Replace `policy_generate_action` with model generation.
  3. Use `agent_valley_reward` as the reward function passed to GRPOTrainer.
  4. Save LoRA/QLoRA adapters and evaluate with baseline_eval.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.environment import AgentValleyEnv
from env.schemas import Action


def build_prompt(observation: Dict[str, Any]) -> str:
    """Create a compact prompt from the visible observation only."""
    return (
        "You are the policy for AgentValleyEnv. Return only JSON with keys: "
        "primary_action, focus_resource, cooperation_mode, risk_posture, rationale.\n\n"
        f"Visible observation:\n{json.dumps(observation, indent=2)}"
    )


def parse_action(text: str) -> Action:
    """Parse a model completion into a validated Action.

    Invalid or non-JSON completions are converted to a safe fallback action so
    they receive a low but non-crashing reward during training/evaluation.
    """
    try:
        payload = json.loads(text)
        return Action(**payload)
    except Exception:
        return Action(
            primary_action="rest",
            focus_resource="none",
            cooperation_mode="solo",
            risk_posture="cautious",
            rationale="Fallback action after invalid model output format.",
        )


def policy_generate_action(prompt: str) -> str:
    """Small deterministic placeholder policy for dry runs.

    Real GRPO training should replace this function with transformer generation.
    """
    del prompt
    return json.dumps(
        {
            "primary_action": "gather",
            "focus_resource": "food",
            "cooperation_mode": "share",
            "risk_posture": "balanced",
            "rationale": "CPU dry-run policy chooses a safe visible-signal action.",
        }
    )


def run_rollout(completions: Iterable[str], difficulty: str = "easy", seed: int = 42) -> List[float]:
    """Score completions by stepping AgentValleyEnv.

    This mirrors the shape expected by TRL reward functions: a list of generated
    texts goes in, and a list of scalar rewards comes out.
    """
    rewards: List[float] = []
    for index, completion in enumerate(completions):
        env = AgentValleyEnv(difficulty=difficulty, episode_index=index, seed=seed)
        obs = env.reset()
        action = parse_action(completion)
        _next_obs, reward, _done, info = env.step(action)
        # Include a small verifier bonus when the action passes schema and has no
        # format penalty. This preserves the environment reward as source of truth.
        reward_breakdown = info.get("reward_breakdown", {})
        format_penalty = float(reward_breakdown.get("format_penalty", 0.0))
        rewards.append(float(reward) + (0.02 if format_penalty == 0 else 0.0))
        del obs
    return rewards


def agent_valley_reward(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
    """Reward callback compatible with TRL GRPOTrainer-style signatures."""
    del prompts
    difficulty = kwargs.get("difficulty", "easy")
    seed = int(kwargs.get("seed", 42))
    return run_rollout(completions, difficulty=difficulty, seed=seed)


def dry_run() -> Dict[str, Any]:
    env = AgentValleyEnv(difficulty="easy", seed=42)
    obs = env.reset()
    prompt = build_prompt(obs)
    completion = policy_generate_action(prompt)
    reward = agent_valley_reward([prompt], [completion])[0]
    return {
        "prompt_preview": prompt[:300],
        "completion": json.loads(completion),
        "reward": round(reward, 4),
        "note": "Dry run passed. Replace policy_generate_action with model generation for real GRPO.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentValleyEnv GRPO/TRL reward bridge")
    parser.add_argument("--dry-run", action="store_true", help="Run a CPU-safe reward bridge check")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(dry_run(), indent=2))
        return 0

    print(
        "This file is a lightweight TRL/Unsloth integration bridge. "
        "Run with --dry-run for a CPU-safe check. For full GRPO training, install "
        "torch, transformers, trl, and unsloth, then pass agent_valley_reward to GRPOTrainer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
