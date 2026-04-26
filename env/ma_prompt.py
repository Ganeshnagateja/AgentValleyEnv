"""Prompt builder for LLM agents acting in MultiAgentValleyEnv."""

from __future__ import annotations

from env.agents import AGENT_SPECS
from env.schemas import CooperationMode, ResourceFocus, RiskPosture, ValleyAction


def _pct(value: float) -> str:
    return f"{float(value) * 100:.0f}%"


def build_agent_prompt(obs: dict) -> str:
    """Build a concise JSON-action prompt from a partial observation."""
    agent_id = str(obs.get("agent_id", "unknown"))
    spec = AGENT_SPECS.get(agent_id)
    role = str(obs.get("agent_role", spec.role.value if spec else "unknown"))
    home_region = str(obs.get("agent_home_region", spec.home_region if spec else "unknown"))
    preferred_action = str(obs.get("preferred_action", spec.preferred_action if spec else "cooperate"))
    preferred_focus = str(obs.get("preferred_focus", spec.preferred_focus if spec else "none"))

    return f"""You are Agent Valley agent '{agent_id}'.
Role: {role}
Home region: {home_region}
Scenario: {obs.get('scenario', 'unknown')}
Task goal: {obs.get('task_goal', 'stabilize the valley')}

Visible supplies:
- food: {_pct(obs.get('food_supply', 0.0))}
- wood: {_pct(obs.get('wood_supply', 0.0))}
- stone: {_pct(obs.get('stone_supply', 0.0))}
- gold: {_pct(obs.get('gold_supply', 0.0))}
- ore: {_pct(obs.get('ore_supply', 0.0))}

Global signals:
- threat level: {_pct(obs.get('threat_level', 0.0))}
- cooperation index: {_pct(obs.get('cooperation_index', 0.0))}
- defense readiness: {_pct(obs.get('defense_readiness', 0.0))}
- average energy: {_pct(obs.get('average_energy', 0.0))}
- average health: {_pct(obs.get('average_health', 0.0))}
- market volatility: {_pct(obs.get('market_volatility', 0.0))}

Role hint: you usually help by choosing primary_action='{preferred_action}' and focus_resource='{preferred_focus}', but adapt to the visible state and coordinate with the team.

Respond ONLY with a JSON object with keys:
primary_action, focus_resource, cooperation_mode, risk_posture, rationale

Valid primary_action values: {[item.value for item in ValleyAction]}
Valid focus_resource values: {[item.value for item in ResourceFocus]}
Valid cooperation_mode values: {[item.value for item in CooperationMode]}
Valid risk_posture values: {[item.value for item in RiskPosture]}

The rationale should explain the decision in one sentence.

Example:
{{"primary_action":"defend","focus_resource":"none","cooperation_mode":"protect","risk_posture":"cautious","rationale":"Threat is high, so I will protect the team while others gather and build."}}"""
