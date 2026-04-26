"""Numeric feature encoding for lightweight neural policies."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


DIFFICULTIES = ("easy", "medium", "hard")
REGIONS = ("farmland", "mine", "forest", "village", "wilderness", "plains", "river", "hills", "coast")
NUMERIC_FIELDS = (
    "tick",
    "active_agents",
    "food_supply",
    "wood_supply",
    "stone_supply",
    "gold_supply",
    "ore_supply",
    "average_health",
    "average_energy",
    "cooperation_index",
    "threat_level",
    "market_volatility",
    "event_severity",
    "region_danger",
    "defense_readiness",
)


def _one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _hash_text(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def encode_observation(observation: Mapping[str, Any]) -> list[float]:
    """Convert an agent-visible observation into deterministic features."""
    features: list[float] = []
    features.extend(_one_hot(str(observation.get("difficulty", "")), DIFFICULTIES))
    features.extend(_one_hot(str(observation.get("region", "")), REGIONS))

    for field in NUMERIC_FIELDS:
        value = float(observation.get(field, 0.0))
        if field == "tick":
            value = value / 10.0
        elif field == "active_agents":
            value = value / 16.0
        features.append(value)

    features.append(_hash_text(str(observation.get("scenario", ""))))
    features.append(_hash_text(str(observation.get("task_goal", ""))))
    return features


FEATURE_DIM = len(encode_observation({}))


def discretize_observation(observation: Mapping[str, Any], bins: int = 5) -> str:
    """Encode an observation into a deterministic tabular state key."""
    parts: list[str] = [
        f"d={observation.get('difficulty', '')}",
        f"t={int(observation.get('tick', 0))}",
        f"r={observation.get('region', '')}",
        f"a={int(observation.get('active_agents', 0))}",
    ]
    for field in NUMERIC_FIELDS:
        if field in {"tick", "active_agents"}:
            continue
        value = max(0.0, min(1.0, float(observation.get(field, 0.0))))
        bucket = min(bins - 1, int(value * bins))
        parts.append(f"{field[:4]}={bucket}")
    return "|".join(parts)
