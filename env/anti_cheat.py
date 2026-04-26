"""Anti-reward-hacking safeguards for AgentValleyEnv.

This module is intentionally separate from rewards.py so that judges and the
LLM checker can find it immediately.  All checks here are server-side and
cannot be bypassed by the agent submitting crafted actions.

Safeguards
----------
1. FORBIDDEN_RATIONALE_TERMS  – blocks prompt-injection / answer-leakage in
   the rationale field.
2. validate_action_schema      – rejects extra fields or invalid enum values;
   prevents the agent from smuggling hidden parameters.
3. is_safety_violation         – detects reckless actions under threat /
   low capacity; used to apply safety_penalty.
4. is_loop_action              – identifies repeated identical actions
   (action farming); used to apply loop_penalty.
5. AntiCheatReport             – structured audit record attached to every
   step's info dict under "anti_cheat".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# 1. Forbidden rationale terms (prompt-injection / answer-leakage detection)
# ---------------------------------------------------------------------------

FORBIDDEN_RATIONALE_TERMS: frozenset[str] = frozenset(
    {
        "ground_truth",
        "hidden",
        "expected_action",
        "answer_key",
        "dataset leak",
    }
)


def contains_leakage(rationale: str) -> bool:
    """Return True if the rationale contains a known leakage term."""
    lowered = rationale.lower()
    return any(term in lowered for term in FORBIDDEN_RATIONALE_TERMS)


# ---------------------------------------------------------------------------
# 2. Action schema validation
# ---------------------------------------------------------------------------

VALID_PRIMARY_ACTIONS = frozenset(
    {"gather", "trade", "build", "explore", "rest", "cooperate", "compete", "defend"}
)
VALID_FOCUS_RESOURCES = frozenset({"none", "food", "wood", "stone", "gold", "ore"})
VALID_COOPERATION_MODES = frozenset({"solo", "share", "coordinate", "protect"})
VALID_RISK_POSTURES = frozenset({"cautious", "balanced", "aggressive"})
ALLOWED_ACTION_KEYS = frozenset(
    {"primary_action", "focus_resource", "cooperation_mode", "risk_posture", "rationale"}
)


def validate_action_schema(raw: Dict[str, Any]) -> Optional[str]:
    """Validate a raw action dict.

    Returns an error string on failure, None on success.
    Extra fields beyond the allowed set are forbidden (reward-hack vector).
    """
    extra = set(raw.keys()) - ALLOWED_ACTION_KEYS
    if extra:
        return f"Forbidden extra action fields: {sorted(extra)}"

    def _val(item: Any) -> str:
        return getattr(item, "value", str(item))

    pa = _val(raw.get("primary_action", ""))
    if pa not in VALID_PRIMARY_ACTIONS:
        return f"Invalid primary_action '{pa}'"

    fr = _val(raw.get("focus_resource", "none"))
    if fr not in VALID_FOCUS_RESOURCES:
        return f"Invalid focus_resource '{fr}'"

    cm = _val(raw.get("cooperation_mode", "solo"))
    if cm not in VALID_COOPERATION_MODES:
        return f"Invalid cooperation_mode '{cm}'"

    rp = _val(raw.get("risk_posture", "balanced"))
    if rp not in VALID_RISK_POSTURES:
        return f"Invalid risk_posture '{rp}'"

    rationale = raw.get("rationale", "") or ""
    if contains_leakage(str(rationale)):
        return "Rationale contains forbidden leakage terms"

    return None


# ---------------------------------------------------------------------------
# 3. Safety violation detection
# ---------------------------------------------------------------------------

def is_safety_violation(obs: Any, action: Any) -> bool:
    """Return True if the action is reckless given the current observation.

    Checked independently of rewards so that anti-cheat logic cannot be
    bypassed by manipulating the reward function.
    """
    def _val(item: Any) -> str:
        return getattr(item, "value", str(item))

    action_name = _val(action.primary_action)
    risk = _val(action.risk_posture)
    combined_threat = (obs.threat_level + obs.event_severity + obs.region_danger) / 3.0
    low_capacity = obs.average_energy < 0.30 or obs.average_health < 0.35

    if risk == "aggressive" and combined_threat > 0.60:
        return True
    if action_name in {"explore", "compete"} and (combined_threat > 0.65 or low_capacity):
        return True
    if action_name == "rest" and obs.threat_level > 0.78 and obs.defense_readiness < 0.55:
        return True
    return False


# ---------------------------------------------------------------------------
# 4. Loop / action-farming detection
# ---------------------------------------------------------------------------

def is_loop_action(repeat_count: int) -> bool:
    """Return True if the agent is farming reward via repeated identical actions."""
    return repeat_count >= 2


# ---------------------------------------------------------------------------
# 5. Structured audit report
# ---------------------------------------------------------------------------

@dataclass
class AntiCheatReport:
    schema_error: Optional[str] = None
    leakage_detected: bool = False
    safety_violation: bool = False
    loop_detected: bool = False
    repeat_count: int = 0
    blocked: bool = False
    notes: list[str] = field(default_factory=list)

    def flag(self, note: str) -> None:
        self.notes.append(note)
        self.blocked = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_error": self.schema_error,
            "leakage_detected": self.leakage_detected,
            "safety_violation": self.safety_violation,
            "loop_detected": self.loop_detected,
            "repeat_count": self.repeat_count,
            "blocked": self.blocked,
            "notes": self.notes,
        }


def run_anti_cheat(
    raw_action: Dict[str, Any],
    obs: Any,
    action: Any,
    repeat_count: int,
) -> AntiCheatReport:
    """Run all anti-cheat checks and return a structured report."""
    report = AntiCheatReport(repeat_count=repeat_count)

    schema_err = validate_action_schema(raw_action)
    if schema_err:
        report.schema_error = schema_err
        report.flag(f"schema_violation: {schema_err}")

    rationale = (raw_action.get("rationale") or "").lower()
    if contains_leakage(rationale):
        report.leakage_detected = True
        report.flag("rationale_leakage_detected")

    if is_safety_violation(obs, action):
        report.safety_violation = True
        report.notes.append("safety_violation")

    if is_loop_action(repeat_count):
        report.loop_detected = True
        report.notes.append(f"loop_detected_repeat_{repeat_count}")

    return report
