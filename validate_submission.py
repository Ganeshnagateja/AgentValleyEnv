"""Local submission sanity checks for AgentValleyEnv."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = [
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt",
    "baseline_eval.py",
    "inference.py",
    "env/environment.py",
    "env/agents.py",
    "env/multi_agent_env.py",
    "env/schemas.py",
    "env/rewards.py",
    "env/graders.py",
    "env/tasks.py",
    "training/ma_grpo_train.py",
    "datasets/easy.json",
    "datasets/medium.json",
    "datasets/hard.json",
    "server/app.py",
    "assets/reward_curve.png",
    "assets/loss_curve.png",
    "scripts/generate_training_plots.py",
    "docs/writeup.md",
    "notebooks/agent_valley_training.ipynb",
    "notebooks/multi_agent_training.ipynb",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    env_module = importlib.import_module("env.environment")
    env = env_module.AgentValleyEnv(difficulty="easy", seed=42)
    obs = env.reset()
    assert isinstance(obs, dict), "reset() must return dict observation"
    action = {
        "primary_action": "gather",
        "focus_resource": "food",
        "cooperation_mode": "share",
        "risk_posture": "balanced",
        "rationale": "baseline visible-signal resource action",
    }
    next_obs, reward, done, info = env.step(action)
    assert isinstance(next_obs, dict), "step() observation must be dict"
    assert isinstance(reward, float), "step() reward must be float"
    assert isinstance(done, bool), "step() done must be bool"
    assert isinstance(info, dict), "step() info must be dict"
    assert isinstance(env.state(), dict), "state() must return dict"

    ma_module = importlib.import_module("env.multi_agent_env")
    ma_env = ma_module.MultiAgentValleyEnv(difficulty="easy", seed=42)
    ma_obs = ma_env.reset()
    assert set(ma_obs) == {"farmer", "miner", "builder", "warrior"}, "multi-agent reset must return four agents"
    ma_actions = {
        "farmer": {"primary_action": "gather", "focus_resource": "food", "cooperation_mode": "share", "risk_posture": "balanced", "rationale": "Farmer gathers food for team survival."},
        "miner": {"primary_action": "gather", "focus_resource": "ore", "cooperation_mode": "coordinate", "risk_posture": "balanced", "rationale": "Miner gathers ore for coordinated work."},
        "builder": {"primary_action": "build", "focus_resource": "stone", "cooperation_mode": "coordinate", "risk_posture": "cautious", "rationale": "Builder improves shared infrastructure."},
        "warrior": {"primary_action": "defend", "focus_resource": "none", "cooperation_mode": "protect", "risk_posture": "cautious", "rationale": "Warrior defends the team from threats."},
    }
    _ma_next_obs, ma_rewards, ma_dones, ma_info = ma_env.step(ma_actions)
    assert set(ma_rewards) == {"farmer", "miner", "builder", "warrior"}, "multi-agent rewards must cover all agents"
    assert "cooperation_bonus" in ma_info, "multi-agent info must include cooperation bonus"

    result = subprocess.run(
        [sys.executable, "baseline_eval.py", "--no-llm", "--episodes", "1"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return result.returncode

    lines = result.stdout.strip().splitlines()
    json_start = max(index for index, line in enumerate(lines) if line.strip() == "{")
    summary = json.loads("\n".join(lines[json_start:]))
    if summary["average"] <= 0:
        print("Baseline average must be positive")
        return 1

    print("Local AgentValleyEnv submission checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
