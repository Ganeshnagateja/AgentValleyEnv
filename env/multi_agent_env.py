"""Multi-agent Agent Valley environment for Theme #1 submissions."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from env.action_space import validate_action
from env.agents import AGENT_IDS, AGENT_SPECS, get_partial_obs
from env.environment import AgentValleyEnv
from env.openenv_compat import Environment
from env.rewards import compute_conflict_penalty, compute_cooperation_bonus


class MultiAgentValleyEnv(Environment):
    """Four-role simultaneous-action wrapper around AgentValleyEnv.

    Agents submit actions at the same tick. A deterministic lead-agent rule
    chooses which action advances the underlying benchmark environment, while
    all agents receive individual role reward plus shared cooperation/conflict
    terms. This preserves the single-agent OpenEnv-compatible environment while
    exposing real multi-agent interaction mechanics.
    """

    name = "MultiAgentValleyEnv"
    version = "1.0.0"
    description = "Four-agent cooperative resource-management environment with partial observability"

    def __init__(self, difficulty: str = "hard", episode_index: int = 0, seed: int = 42):
        self.difficulty = difficulty
        self.episode_index = episode_index
        self.seed_value = seed
        self.inner_env = AgentValleyEnv(difficulty=difficulty, episode_index=episode_index, seed=seed)
        self._step_idx = 0
        self._done = True
        self._last_shared_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}

    def available_agents(self) -> list[str]:
        return list(AGENT_IDS)

    def reset(self, episode_index: int | None = None, seed: int | None = None) -> Dict[str, dict]:
        if episode_index is not None:
            self.episode_index = episode_index
        if seed is not None:
            self.seed_value = seed

        self.inner_env = AgentValleyEnv(
            difficulty=self.difficulty,
            episode_index=self.episode_index,
            seed=self.seed_value,
        )
        shared_obs = self.inner_env.reset(episode_index=self.episode_index, seed=self.seed_value)
        self._last_shared_obs = dict(shared_obs)
        self._step_idx = 0
        self._done = False
        self._last_info = {}
        return self._partial_observations(shared_obs)

    def step(self, actions: Dict[str, dict]) -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        missing = [agent_id for agent_id in AGENT_IDS if agent_id not in actions]
        if missing:
            raise ValueError(f"Missing actions for agents: {missing}")
        unexpected = [agent_id for agent_id in actions if agent_id not in AGENT_IDS]
        if unexpected:
            raise ValueError(f"Unknown agent IDs: {unexpected}")

        shared_obs = dict(self._last_shared_obs or {})
        validated_actions = {
            agent_id: validate_action(actions[agent_id]).model_dump(mode="json")
            for agent_id in AGENT_IDS
        }
        lead_agent = self._pick_lead_agent(shared_obs, validated_actions)
        next_shared_obs, base_env_reward, done, inner_info = self.inner_env.step(validated_actions[lead_agent])

        role_rewards = {
            agent_id: self._compute_role_reward(agent_id, validated_actions[agent_id], shared_obs)
            for agent_id in AGENT_IDS
        }
        cooperation_bonus = self._compute_cooperation_bonus(validated_actions, shared_obs)
        conflict_penalty = self._compute_conflict_penalty(validated_actions, shared_obs)

        reward_components_by_agent: dict[str, dict[str, float]] = {}
        rewards: dict[str, float] = {}
        for agent_id in AGENT_IDS:
            base_component = float(base_env_reward) if agent_id == lead_agent else 0.0
            total = role_rewards[agent_id] + base_component + cooperation_bonus + conflict_penalty
            clipped_total = max(-1.0, min(1.0, total))
            rewards[agent_id] = round(clipped_total, 4)
            reward_components_by_agent[agent_id] = {
                "role_reward": round(role_rewards[agent_id], 4),
                "base_env_reward": round(base_component, 4),
                "cooperation_bonus": round(cooperation_bonus, 4),
                "conflict_penalty": round(conflict_penalty, 4),
                "total": rewards[agent_id],
            }

        self._step_idx += 1
        self._done = bool(done)
        self._last_shared_obs = dict(next_shared_obs)
        dones = {agent_id: self._done for agent_id in AGENT_IDS}
        dones["__all__"] = self._done
        info: dict[str, Any] = {
            "multi_agent": True,
            "step": self._step_idx,
            "lead_agent": lead_agent,
            "cooperation_bonus": round(cooperation_bonus, 4),
            "conflict_penalty": round(conflict_penalty, 4),
            "individual_role_rewards": {key: round(value, 4) for key, value in role_rewards.items()},
            "base_env_reward": round(float(base_env_reward), 4),
            "reward_components_by_agent": reward_components_by_agent,
            "shared_observation": next_shared_obs,
            "inner_info": inner_info,
        }
        if self._done:
            info["episode_result"] = inner_info.get("episode_result")
        self._last_info = info
        return self._partial_observations(next_shared_obs), rewards, dones, info

    def state(self) -> dict:
        return {
            "multi_agent": True,
            "name": self.name,
            "difficulty": self.difficulty,
            "seed": self.seed_value,
            "episode_index": self.episode_index,
            "step_idx": self._step_idx,
            "done": self._done,
            "agents": {
                agent_id: {
                    "role": spec.role.value,
                    "home_region": spec.home_region,
                    "preferred_action": spec.preferred_action,
                    "preferred_focus": spec.preferred_focus,
                    "preferred_cooperation": spec.preferred_cooperation,
                }
                for agent_id, spec in AGENT_SPECS.items()
            },
            "inner_state": self.inner_env.state(),
            "last_info": self._last_info,
        }

    def agent_observation(self, agent_id: str) -> dict:
        if self._last_shared_obs is None:
            raise RuntimeError("Environment has not been reset.")
        return get_partial_obs(self._last_shared_obs, agent_id, self.seed_value + self._step_idx)

    def render(self) -> str:
        state = self.state()
        return (
            f"MultiAgentValleyEnv({self.difficulty}) step={state['step_idx']} "
            f"agents={','.join(AGENT_IDS)} done={state['done']}"
        )

    def _partial_observations(self, shared_obs: dict) -> Dict[str, dict]:
        return {
            agent_id: get_partial_obs(shared_obs, agent_id, self.seed_value + self._step_idx)
            for agent_id in AGENT_IDS
        }

    def _pick_lead_agent(self, obs: dict, actions: Dict[str, dict]) -> str:
        if float(obs.get("threat_level", 0.0)) > 0.65:
            return "warrior"
        if float(obs.get("food_supply", 1.0)) < 0.25:
            return "farmer"
        if float(obs.get("defense_readiness", 1.0)) < 0.35:
            return "builder"
        return AGENT_IDS[self._step_idx % len(AGENT_IDS)]

    def _compute_role_reward(self, agent_id: str, action_dict: dict, obs: dict) -> float:
        spec = AGENT_SPECS[agent_id]
        action = validate_action(action_dict)
        reward = 0.0
        if str(action.primary_action) == spec.preferred_action:
            reward += 0.20
        if str(action.focus_resource) == spec.preferred_focus:
            reward += 0.12
        if str(action.cooperation_mode) == spec.preferred_cooperation:
            reward += 0.08
        return round(reward, 4)

    def _compute_cooperation_bonus(self, actions_dict: Dict[str, dict], obs: dict) -> float:
        return compute_cooperation_bonus(actions_dict, obs)

    def _compute_conflict_penalty(self, actions_dict: Dict[str, dict], obs: dict) -> float:
        return compute_conflict_penalty(actions_dict, obs)
