"""OpenEnv-compliant Agent Valley environment."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from env.graders import grade_episode
from env.openenv_compat import Environment
from env.rewards import compute_step_reward
from env.schemas import (
    Action,
    CooperationMode,
    EpisodeResult,
    FinalStatus,
    Observation,
    ResourceFocus,
    RewardPayload,
    RiskPosture,
    ValleyAction,
    ValleyRegion,
)
from env.tasks import TaskConfig, get_task, load_episode


ACTION_SPACE: Dict[str, Any] = {
    "type": "dict",
    "fields": {
        "primary_action": {"type": "categorical", "values": [item.value for item in ValleyAction]},
        "focus_resource": {"type": "categorical", "values": [item.value for item in ResourceFocus]},
        "cooperation_mode": {"type": "categorical", "values": [item.value for item in CooperationMode]},
        "risk_posture": {"type": "categorical", "values": [item.value for item in RiskPosture]},
        "rationale": {"type": "string", "max_length": 512, "required": False},
    },
}


OBSERVATION_SPACE: Dict[str, Any] = {
    "type": "dict",
    "fields": {
        "tick": {"type": "int", "range": [0, 10_000]},
        "difficulty": {"type": "categorical", "values": ["easy", "medium", "hard"]},
        "scenario": {"type": "string"},
        "task_goal": {"type": "string"},
        "region": {"type": "categorical", "values": [item.value for item in ValleyRegion]},
        "active_agents": {"type": "int", "range": [1, 16]},
        "food_supply": {"type": "float", "range": [0.0, 1.0]},
        "wood_supply": {"type": "float", "range": [0.0, 1.0]},
        "stone_supply": {"type": "float", "range": [0.0, 1.0]},
        "gold_supply": {"type": "float", "range": [0.0, 1.0]},
        "ore_supply": {"type": "float", "range": [0.0, 1.0]},
        "average_health": {"type": "float", "range": [0.0, 1.0]},
        "average_energy": {"type": "float", "range": [0.0, 1.0]},
        "cooperation_index": {"type": "float", "range": [0.0, 1.0]},
        "threat_level": {"type": "float", "range": [0.0, 1.0]},
        "market_volatility": {"type": "float", "range": [0.0, 1.0]},
        "event_severity": {"type": "float", "range": [0.0, 1.0]},
        "region_danger": {"type": "float", "range": [0.0, 1.0]},
        "defense_readiness": {"type": "float", "range": [0.0, 1.0]},
    },
}


class AgentValleyEnv(Environment):
    """Verifiable multi-agent valley management benchmark.

    Required OpenEnv interface:
      reset() -> dict
      step(action) -> (dict, float, bool, dict)
      state() -> dict
    """

    name = "AgentValleyEnv"
    version = "1.0.0"
    description = "Multi-agent RL environment for valley survival, cooperation, and crisis response"

    action_space = ACTION_SPACE
    observation_space = OBSERVATION_SPACE

    def __init__(
        self,
        difficulty: str = "easy",
        episode_index: int = 0,
        seed: int = 42,
        max_steps: Optional[int] = None,
    ):
        self._difficulty = difficulty
        self._episode_index = episode_index
        self._seed = seed
        self._task_cfg: TaskConfig = get_task(difficulty)
        self._max_steps = max_steps or self._task_cfg.max_steps

        self._episode_id = ""
        self._observations: List[Observation] = []
        self._trajectory: List[Tuple[Observation, Action, RewardPayload]] = []
        self._step_idx = 0
        self._cumulative_reward = 0.0
        self._prev_action: Optional[Action] = None
        self._repeat_count = 0
        self._done = True

    def reset(self, episode_index: Optional[int] = None, seed: Optional[int] = None) -> dict:
        if episode_index is not None:
            self._episode_index = episode_index
        if seed is not None:
            self._seed = seed

        self._episode_id = str(uuid.uuid4())[:8]
        self._observations = load_episode(self._difficulty, self._episode_index, self._seed)
        self._trajectory = []
        self._step_idx = 0
        self._cumulative_reward = 0.0
        self._prev_action = None
        self._repeat_count = 0
        self._done = False
        return self._observations[0].agent_view()

    def step(self, action: dict | Action) -> Tuple[dict, float, bool, dict]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")
        if isinstance(action, dict):
            action = Action(**action)

        obs = self._observations[self._step_idx]

        if (
            self._prev_action
            and action.primary_action == self._prev_action.primary_action
            and action.focus_resource == self._prev_action.focus_resource
            and action.cooperation_mode == self._prev_action.cooperation_mode
        ):
            self._repeat_count += 1
        else:
            self._repeat_count = 0
        self._prev_action = action

        reward_payload = compute_step_reward(
            obs=obs,
            action=action,
            step_idx=self._step_idx,
            repeat_count=self._repeat_count,
        )
        reward = round(reward_payload.total, 4)
        self._cumulative_reward += reward
        self._trajectory.append((obs, action, reward_payload))

        self._step_idx += 1
        done = self._step_idx >= len(self._observations) or self._step_idx >= self._max_steps
        self._done = done

        next_obs = obs.agent_view() if done else self._observations[self._step_idx].agent_view()
        info: Dict[str, Any] = {
            "step": self._step_idx,
            "max_steps": self._max_steps,
            "reward_breakdown": reward_payload.model_dump(mode="json"),
            "repeat_count": self._repeat_count,
        }

        if done:
            grader_result = grade_episode(self._trajectory, self._difficulty)
            totals = self._reward_component_totals()
            result = EpisodeResult(
                episode_id=self._episode_id,
                task_difficulty=self._difficulty,
                scenario=self._observations[0].scenario,
                final_status=grader_result.final_status,
                total_steps=self._step_idx,
                cumulative_reward=round(self._cumulative_reward, 4),
                task_score=grader_result.score,
                predicted_actions=[item[1].primary_action for item in self._trajectory],
                expected_actions=[item[0].ground_truth_action or "" for item in self._trajectory],
                predicted_focus=[item[1].focus_resource for item in self._trajectory],
                expected_focus=[item[0].ground_truth_focus or "" for item in self._trajectory],
                score_components=grader_result.components,
                reward_totals=totals,
            )
            info["episode_result"] = result.model_dump(mode="json")
            info["grader_details"] = grader_result.details

        return next_obs, reward, done, info

    def state(self) -> dict:
        return {
            "episode_id": self._episode_id,
            "difficulty": self._difficulty,
            "scenario": self._observations[0].scenario if self._observations else None,
            "step_idx": self._step_idx,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "trajectory_length": len(self._trajectory),
            "seed": self._seed,
            "episode_index": self._episode_index,
        }

    def render(self) -> str:
        if not self._observations:
            return "AgentValleyEnv has not been reset."
        status = FinalStatus.VALLEY_COLLAPSED.value
        if self._done and self._trajectory:
            status = grade_episode(self._trajectory, self._difficulty).final_status.value
        return (
            f"AgentValleyEnv({self._difficulty}) scenario={self._observations[0].scenario} "
            f"step={self._step_idx}/{self._max_steps} reward={self._cumulative_reward:.3f} "
            f"status={status if self._done else 'running'}"
        )

    def seed(self, value: int) -> None:
        self._seed = value

    def _reward_component_totals(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for _obs, _action, reward in self._trajectory:
            for key, value in reward.model_dump(mode="json").items():
                if key == "total":
                    continue
                totals[key] = totals.get(key, 0.0) + float(value)
        return {key: round(value, 4) for key, value in sorted(totals.items())}
