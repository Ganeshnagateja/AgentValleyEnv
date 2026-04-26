"""Discrete composite action space for AgentValleyEnv.

The environment action is a structured tuple:
primary_action, focus_resource, cooperation_mode, risk_posture, rationale.
Only the finite tuple fields are indexed.  Rationales are generated safely when
an indexed action is materialized for env.step().
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any, Mapping, Sequence

from env.schemas import Action, CooperationMode, ResourceFocus, RiskPosture, ValleyAction


ActionTuple = tuple[str, str, str, str]


def default_rationale(
    primary_action: str,
    focus_resource: str,
    cooperation_mode: str,
    risk_posture: str,
) -> str:
    focus = "no resource" if focus_resource == "none" else f"{focus_resource} resource"
    return (
        f"Choose {primary_action} with {focus}, {cooperation_mode} coordination, "
        f"and a {risk_posture} risk posture from visible valley signals."
    )


def validate_action(action: Action | Mapping[str, Any]) -> Action:
    """Return a schema-validated Action with a safe rationale if missing."""
    if isinstance(action, Action):
        data = action.model_dump(mode="json")
    else:
        data = dict(action)

    primary = str(data.get("primary_action", ""))
    focus = str(data.get("focus_resource", ResourceFocus.NONE.value))
    cooperation = str(data.get("cooperation_mode", CooperationMode.SOLO.value))
    risk = str(data.get("risk_posture", RiskPosture.BALANCED.value))
    if not data.get("rationale"):
        data["rationale"] = default_rationale(primary, focus, cooperation, risk)
    return Action(**data)


def _tuple_from_action(action: Action | Mapping[str, Any]) -> ActionTuple:
    validated = validate_action(action)
    return (
        str(validated.primary_action),
        str(validated.focus_resource),
        str(validated.cooperation_mode),
        str(validated.risk_posture),
    )


@lru_cache(maxsize=1)
def _indexed_actions() -> tuple[Action, ...]:
    actions: list[Action] = []
    for primary in ValleyAction:
        for focus in ResourceFocus:
            for cooperation in CooperationMode:
                for risk in RiskPosture:
                    rationale = default_rationale(primary.value, focus.value, cooperation.value, risk.value)
                    try:
                        actions.append(
                            Action(
                                primary_action=primary.value,
                                focus_resource=focus.value,
                                cooperation_mode=cooperation.value,
                                risk_posture=risk.value,
                                rationale=rationale,
                            )
                        )
                    except ValueError:
                        # The Action schema rejects invalid combinations, such
                        # as defend/rest with ore or gold focus.
                        continue
    return tuple(actions)


@lru_cache(maxsize=1)
def _index_by_tuple() -> dict[ActionTuple, int]:
    return {_tuple_from_action(action): index for index, action in enumerate(_indexed_actions())}


def action_count() -> int:
    return len(_indexed_actions())


def list_all_actions() -> list[dict[str, Any]]:
    """Return every valid composite action as JSON-serializable dicts."""
    return [action.model_dump(mode="json") for action in _indexed_actions()]


def action_to_index(action: Action | Mapping[str, Any]) -> int:
    """Map a schema-valid composite action to its deterministic index."""
    key = _tuple_from_action(action)
    try:
        return _index_by_tuple()[key]
    except KeyError as exc:
        raise ValueError(f"Action is outside the indexed action space: {key}") from exc


def index_to_action(index: int) -> Action:
    """Map an action index back to a schema-valid Action."""
    actions = _indexed_actions()
    if index < 0 or index >= len(actions):
        raise IndexError(f"Action index {index} out of range 0..{len(actions) - 1}")
    action = actions[index]
    return Action(**action.model_dump(mode="json"))


def random_action(rng: random.Random | None = None) -> Action:
    """Sample a valid action using an optional seeded random generator."""
    generator = rng or random
    actions: Sequence[Action] = _indexed_actions()
    return index_to_action(generator.randrange(len(actions)))
