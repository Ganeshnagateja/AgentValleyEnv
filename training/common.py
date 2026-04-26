"""Shared training utilities for AgentValleyEnv."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
DIFFICULTIES = ("easy", "medium", "hard")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def project_relative(path: Path | str) -> str:
    """Return a stable repo-relative POSIX path for metrics/API payloads."""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def resolve_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows[-limit:] if limit and limit > 0 else rows


def reset_file(path: Path) -> None:
    ensure_parent(path)
    path.write_text("", encoding="utf-8")


def resolve_difficulty(configured: str, episode: int) -> str:
    if configured == "mixed":
        return DIFFICULTIES[(episode - 1) % len(DIFFICULTIES)]
    if configured not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty '{configured}'")
    return configured


def score_components(episode_result: dict[str, Any]) -> dict[str, float | None]:
    components = episode_result.get("score_components") or {}
    return {
        "action_accuracy": components.get("action_accuracy"),
        "safety_score": components.get("safety_score"),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)
