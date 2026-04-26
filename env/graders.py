"""Deterministic episode graders for AgentValleyEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from env.rewards import _is_safety_violation, _value
from env.schemas import Action, FinalStatus, Observation, RewardPayload


@dataclass(frozen=True)
class GraderResult:
    score: float
    final_status: FinalStatus
    components: Dict[str, float]
    details: Dict[str, object]


Trajectory = List[Tuple[Observation, Action, RewardPayload]]


def _mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def _match(value, expected) -> float:
    return 1.0 if _value(value) == _value(expected) else 0.0


def grade_episode(trajectory: Trajectory, difficulty: str) -> GraderResult:
    if not trajectory:
        return GraderResult(
            score=0.0,
            final_status=FinalStatus.VALLEY_COLLAPSED,
            components={},
            details={"reason": "empty trajectory"},
        )

    action_scores: List[float] = []
    focus_scores: List[float] = []
    cooperation_scores: List[float] = []
    risk_scores: List[float] = []
    safety_violations = 0

    for obs, action, _reward in trajectory:
        action_name = _value(action.primary_action)
        expected_action = _value(obs.ground_truth_action)
        acceptable = [_value(item) for item in (obs.acceptable_actions or [])]
        if expected_action and expected_action not in acceptable:
            acceptable.append(expected_action)

        if action_name == expected_action:
            action_scores.append(1.0)
        elif action_name in acceptable:
            action_scores.append(0.65)
        else:
            action_scores.append(0.0)

        focus_scores.append(_match(action.focus_resource, obs.ground_truth_focus))
        cooperation_scores.append(_match(action.cooperation_mode, obs.ground_truth_cooperation))
        risk_scores.append(_match(action.risk_posture, obs.ground_truth_risk))
        if _is_safety_violation(obs, action):
            safety_violations += 1

    safety_score = max(0.0, 1.0 - safety_violations / max(len(trajectory), 1))
    weights = {
        "easy": (0.56, 0.16, 0.12, 0.08, 0.08),
        "medium": (0.50, 0.14, 0.18, 0.10, 0.08),
        "hard": (0.48, 0.12, 0.18, 0.14, 0.08),
    }.get(difficulty, (0.50, 0.15, 0.15, 0.10, 0.10))

    components = {
        "action_accuracy": round(_mean(action_scores), 4),
        "resource_focus_accuracy": round(_mean(focus_scores), 4),
        "cooperation_alignment": round(_mean(cooperation_scores), 4),
        "risk_alignment": round(_mean(risk_scores), 4),
        "safety_score": round(safety_score, 4),
    }

    score = round(
        weights[0] * components["action_accuracy"]
        + weights[1] * components["resource_focus_accuracy"]
        + weights[2] * components["cooperation_alignment"]
        + weights[3] * components["risk_alignment"]
        + weights[4] * components["safety_score"],
        4,
    )

    if score >= 0.78:
        final_status = FinalStatus.VALLEY_STABILIZED
    elif score >= 0.55:
        final_status = FinalStatus.VALLEY_SURVIVED
    else:
        final_status = FinalStatus.VALLEY_COLLAPSED

    return GraderResult(
        score=score,
        final_status=final_status,
        components=components,
        details={
            "safety_violations": safety_violations,
            "steps_evaluated": len(trajectory),
            "difficulty": difficulty,
        },
    )
