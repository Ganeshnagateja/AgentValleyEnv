"""Dense reward shaping for AgentValleyEnv.

Anti-hacking checks are intentionally delegated to env.anti_cheat so that the
safeguard layer is independently auditable.  This module only computes reward
magnitudes; it never mutates protected environment state.
"""

from __future__ import annotations

from typing import Dict

from env.anti_cheat import is_safety_violation as _is_safety_violation_impl
from env.schemas import Action, Observation, RewardPayload

# Re-export for backward compat (openenv.yaml references env.rewards)
FORBIDDEN_RATIONALE_TERMS = {
    "ground_truth",
    "hidden",
    "expected_action",
    "answer_key",
    "dataset leak",
}


def _value(item) -> str:
    return getattr(item, "value", item)


def _is_safety_violation(obs: Observation, action: Action) -> bool:
    return _is_safety_violation_impl(obs, action)


def compute_step_reward(
    obs: Observation,
    action: Action,
    step_idx: int,
    repeat_count: int,
) -> RewardPayload:
    """Compute a decomposed step reward.

    Components intentionally separate objective success, resource targeting,
    cooperation, safety, loop prevention, and format compliance. This matches
    the hackathon guidance to avoid a single hackable scalar.
    """
    payload = RewardPayload(progress_reward=0.05)

    action_name = _value(action.primary_action)
    expected_action = _value(obs.ground_truth_action)
    acceptable = [_value(item) for item in (obs.acceptable_actions or [])]
    if expected_action and expected_action not in acceptable:
        acceptable.append(expected_action)

    if action_name == expected_action:
        payload.objective_bonus = 0.38
    elif action_name in acceptable:
        payload.objective_bonus = 0.22
    else:
        payload.objective_bonus = -0.18

    focus = _value(action.focus_resource)
    expected_focus = _value(obs.ground_truth_focus)
    if focus == expected_focus:
        payload.focus_bonus = 0.12
    elif expected_focus == "none" and focus in {"food", "wood", "stone"} and action_name in {"gather", "build"}:
        payload.focus_bonus = 0.03
    else:
        payload.focus_bonus = -0.06

    cooperation = _value(action.cooperation_mode)
    expected_cooperation = _value(obs.ground_truth_cooperation)
    if cooperation == expected_cooperation:
        payload.cooperation_bonus = 0.10
    elif cooperation in {"coordinate", "protect", "share"} and obs.cooperation_index < 0.40:
        payload.cooperation_bonus = 0.04
    else:
        payload.cooperation_bonus = -0.04

    risk = _value(action.risk_posture)
    expected_risk = _value(obs.ground_truth_risk)
    if risk == expected_risk:
        payload.cooperation_bonus += 0.04
    elif _is_safety_violation(obs, action):
        payload.safety_penalty -= 0.22
    else:
        payload.safety_penalty -= 0.04

    if _is_safety_violation(obs, action):
        payload.safety_penalty -= 0.25

    if repeat_count >= 2:
        payload.loop_penalty = -0.70
    elif repeat_count == 1 and step_idx > 0:
        payload.loop_penalty = -0.08

    rationale = (action.rationale or "").lower()
    if len(rationale.strip()) < 12:
        payload.format_penalty -= 0.05
    if any(term in rationale for term in FORBIDDEN_RATIONALE_TERMS):
        payload.format_penalty -= 0.30

    return payload.compute_total()


def compute_cooperation_bonus(actions: Dict[str, dict], obs: dict) -> float:
    """
    Compute the shared cooperation bonus for a full set of simultaneous agent actions.

    Called by MultiAgentValleyEnv, not the single-agent env. The bonus rewards
    complementary division of labor rather than every agent taking the same
    locally useful action.
    """
    farmer = actions.get("farmer", {})
    builder = actions.get("builder", {})
    warrior = actions.get("warrior", {})

    def primary(agent_action: dict) -> str:
        return str(agent_action.get("primary_action", ""))

    def focus(agent_action: dict) -> str:
        return str(agent_action.get("focus_resource", "none"))

    def coop(agent_action: dict) -> str:
        return str(agent_action.get("cooperation_mode", "solo"))

    bonus = 0.0
    if primary(warrior) == "defend" and any(
        agent_id != "warrior" and primary(action) != "defend" for agent_id, action in actions.items()
    ):
        bonus += 0.25
    if primary(farmer) == "gather" and focus(farmer) == "food" and primary(builder) == "build":
        bonus += 0.20
    if sum(1 for action in actions.values() if coop(action) in {"coordinate", "protect"}) >= 3:
        bonus += 0.15
    if len({primary(action) for action in actions.values()}) == len(actions) and actions:
        bonus += 0.10
    return round(min(0.50, bonus), 4)


def compute_conflict_penalty(actions: Dict[str, dict], obs: dict) -> float:
    """Compute the shared conflict penalty for simultaneous agent actions."""
    primary_counts: dict[str, int] = {}
    for action in actions.values():
        primary = str(action.get("primary_action", ""))
        primary_counts[primary] = primary_counts.get(primary, 0) + 1

    penalty = 0.0
    if any(count >= 3 for count in primary_counts.values()):
        penalty -= 0.20
    if float(obs.get("threat_level", 0.0)) > 0.75 and str(actions.get("warrior", {}).get("primary_action", "")) != "defend":
        penalty -= 0.15
    if float(obs.get("food_supply", 1.0)) < 0.20 and str(actions.get("farmer", {}).get("primary_action", "")) != "gather":
        penalty -= 0.10
    return round(max(-0.35, penalty), 4)
