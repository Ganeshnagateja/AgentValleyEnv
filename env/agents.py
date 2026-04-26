"""Agent role definitions and partial-observation helpers for Agent Valley."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class AgentRole(str, Enum):
    FARMER = "farmer"
    MINER = "miner"
    BUILDER = "builder"
    WARRIOR = "warrior"


@dataclass(frozen=True)
class AgentSpec:
    role: AgentRole
    home_region: str
    preferred_action: str
    preferred_focus: str
    preferred_cooperation: str


AGENT_SPECS: Dict[str, AgentSpec] = {
    "farmer": AgentSpec(
        role=AgentRole.FARMER,
        home_region="farmland",
        preferred_action="gather",
        preferred_focus="food",
        preferred_cooperation="share",
    ),
    "miner": AgentSpec(
        role=AgentRole.MINER,
        home_region="mine",
        preferred_action="gather",
        preferred_focus="ore",
        preferred_cooperation="coordinate",
    ),
    "builder": AgentSpec(
        role=AgentRole.BUILDER,
        home_region="village",
        preferred_action="build",
        preferred_focus="stone",
        preferred_cooperation="coordinate",
    ),
    "warrior": AgentSpec(
        role=AgentRole.WARRIOR,
        home_region="wilderness",
        preferred_action="defend",
        preferred_focus="none",
        preferred_cooperation="protect",
    ),
}

AGENT_IDS: List[str] = ["farmer", "miner", "builder", "warrior"]

RESOURCE_FIELDS = ("food_supply", "wood_supply", "stone_supply", "gold_supply", "ore_supply")


def _stable_agent_seed(seed: int, agent_id: str) -> int:
    digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) + int(seed)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def get_partial_obs(full_obs: dict, agent_id: str, seed: int = 0) -> dict:
    """Return a deterministic role-specific observation for one agent.

    The dataset has one active region per step. An agent whose home region is
    active sees resource supplies exactly; other agents see noisy resource
    estimates while global threat, cooperation, health, energy, and defense
    fields remain unchanged.
    """
    if agent_id not in AGENT_SPECS:
        raise KeyError(f"Unknown agent_id '{agent_id}'")

    spec = AGENT_SPECS[agent_id]
    obs = dict(full_obs)
    obs["agent_id"] = agent_id
    obs["agent_role"] = spec.role.value
    obs["agent_home_region"] = spec.home_region
    obs["preferred_action"] = spec.preferred_action
    obs["preferred_focus"] = spec.preferred_focus
    obs["preferred_cooperation"] = spec.preferred_cooperation

    current_region = str(full_obs.get("region", ""))
    if current_region != spec.home_region:
        rng = random.Random(_stable_agent_seed(seed, agent_id))
        for field in RESOURCE_FIELDS:
            if field in obs:
                obs[field] = round(_clip01(float(obs[field]) + rng.gauss(0.0, 0.08)), 4)

    return obs
