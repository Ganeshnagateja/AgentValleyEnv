"""Task registry and deterministic dataset loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from env.schemas import Observation


DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


@dataclass(frozen=True)
class TaskConfig:
    difficulty: str
    name: str
    max_steps: int
    dataset: str
    description: str


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        difficulty="easy",
        name="Resource Stabilization",
        max_steps=5,
        dataset="easy.json",
        description="Resolve short-horizon resource and rest decisions before the settlement destabilizes.",
    ),
    "medium": TaskConfig(
        difficulty="medium",
        name="Market Shock Coordination",
        max_steps=7,
        dataset="medium.json",
        description="Coordinate agents through market volatility, scarcity, and moderate external events.",
    ),
    "hard": TaskConfig(
        difficulty="hard",
        name="Invasion Defense and Recovery",
        max_steps=10,
        dataset="hard.json",
        description="Balance defense, recovery, and resource routing during long-horizon valley crises.",
    ),
}


def get_task(difficulty: str) -> TaskConfig:
    try:
        return TASKS[difficulty]
    except KeyError as exc:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of: {valid}") from exc


def _load_raw_episodes(difficulty: str) -> List[dict]:
    task = get_task(difficulty)
    path = DATA_DIR / task.dataset
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    if not episodes:
        raise ValueError(f"Dataset has no episodes: {path}")
    return episodes


def load_episode(difficulty: str, episode_index: int = 0, seed: int = 42) -> List[Observation]:
    """Load one deterministic curriculum episode.

    The seed controls which fixed episode is selected while keeping the steps
    themselves deterministic and inspectable.
    """
    episodes = _load_raw_episodes(difficulty)
    selected = episodes[(episode_index + seed) % len(episodes)]
    return [Observation(**step) for step in selected["steps"]]


def list_tasks() -> Dict[str, dict]:
    return {key: config.__dict__ for key, config in TASKS.items()}
