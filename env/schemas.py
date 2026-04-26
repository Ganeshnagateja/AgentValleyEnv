"""Typed data contracts for the Agent Valley OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ValleyRegion(str, Enum):
    FARMLAND = "farmland"
    MINE = "mine"
    FOREST = "forest"
    VILLAGE = "village"
    WILDERNESS = "wilderness"
    PLAINS = "plains"
    RIVER = "river"
    HILLS = "hills"
    COAST = "coast"


class ValleyAction(str, Enum):
    GATHER = "gather"
    TRADE = "trade"
    BUILD = "build"
    EXPLORE = "explore"
    REST = "rest"
    COOPERATE = "cooperate"
    COMPETE = "compete"
    DEFEND = "defend"


class ResourceFocus(str, Enum):
    NONE = "none"
    FOOD = "food"
    WOOD = "wood"
    STONE = "stone"
    GOLD = "gold"
    ORE = "ore"


class CooperationMode(str, Enum):
    SOLO = "solo"
    SHARE = "share"
    COORDINATE = "coordinate"
    PROTECT = "protect"


class RiskPosture(str, Enum):
    CAUTIOUS = "cautious"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class FinalStatus(str, Enum):
    VALLEY_STABILIZED = "valley_stabilized"
    VALLEY_SURVIVED = "valley_survived"
    VALLEY_COLLAPSED = "valley_collapsed"


class Observation(BaseModel):
    """Agent-visible valley state plus hidden verifier targets."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    tick: int = Field(..., ge=0)
    difficulty: TaskDifficulty
    scenario: str = Field(..., min_length=1, max_length=80)
    task_goal: str = Field(..., min_length=1, max_length=180)
    region: ValleyRegion
    active_agents: int = Field(..., ge=1, le=16)

    food_supply: float = Field(..., ge=0.0, le=1.0)
    wood_supply: float = Field(..., ge=0.0, le=1.0)
    stone_supply: float = Field(..., ge=0.0, le=1.0)
    gold_supply: float = Field(..., ge=0.0, le=1.0)
    ore_supply: float = Field(..., ge=0.0, le=1.0)

    average_health: float = Field(..., ge=0.0, le=1.0)
    average_energy: float = Field(..., ge=0.0, le=1.0)
    cooperation_index: float = Field(..., ge=0.0, le=1.0)
    threat_level: float = Field(..., ge=0.0, le=1.0)
    market_volatility: float = Field(..., ge=0.0, le=1.0)
    event_severity: float = Field(..., ge=0.0, le=1.0)
    region_danger: float = Field(..., ge=0.0, le=1.0)
    defense_readiness: float = Field(..., ge=0.0, le=1.0)

    ground_truth_action: Optional[ValleyAction] = Field(None, exclude=True)
    acceptable_actions: List[ValleyAction] = Field(default_factory=list, exclude=True)
    ground_truth_focus: Optional[ResourceFocus] = Field(None, exclude=True)
    ground_truth_cooperation: Optional[CooperationMode] = Field(None, exclude=True)
    ground_truth_risk: Optional[RiskPosture] = Field(None, exclude=True)

    def agent_view(self) -> dict:
        """Return the observation without hidden answer keys."""
        data = self.model_dump(mode="json")
        for key in (
            "ground_truth_action",
            "acceptable_actions",
            "ground_truth_focus",
            "ground_truth_cooperation",
            "ground_truth_risk",
        ):
            data.pop(key, None)
        return data


class Action(BaseModel):
    """Single agent decision for one valley step."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    primary_action: ValleyAction = Field(..., description="Main action to execute")
    focus_resource: ResourceFocus = Field(ResourceFocus.NONE)
    cooperation_mode: CooperationMode = Field(CooperationMode.SOLO)
    risk_posture: RiskPosture = Field(RiskPosture.BALANCED)
    rationale: Optional[str] = Field(None, max_length=512)

    @field_validator("focus_resource")
    @classmethod
    def rest_and_defend_need_no_resource(cls, value: ResourceFocus, info) -> ResourceFocus:
        action = info.data.get("primary_action")
        action_value = getattr(action, "value", action)
        value_value = getattr(value, "value", value)
        if action_value in {"rest", "defend", "cooperate"} and value_value not in {"none", "food", "wood", "stone"}:
            raise ValueError(f"{action_value} cannot focus on {value_value}")
        return value


class RewardPayload(BaseModel):
    """Interpretable reward components returned on every step."""

    progress_reward: float = 0.0
    objective_bonus: float = 0.0
    focus_bonus: float = 0.0
    cooperation_bonus: float = 0.0
    safety_penalty: float = 0.0
    loop_penalty: float = 0.0
    format_penalty: float = 0.0
    total: float = 0.0

    def compute_total(self) -> "RewardPayload":
        self.total = round(
            self.progress_reward
            + self.objective_bonus
            + self.focus_bonus
            + self.cooperation_bonus
            + self.safety_penalty
            + self.loop_penalty
            + self.format_penalty,
            4,
        )
        return self


class EpisodeResult(BaseModel):
    """Terminal result returned in info['episode_result'] when done is true."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    episode_id: str
    task_difficulty: TaskDifficulty
    scenario: str
    final_status: FinalStatus
    total_steps: int
    cumulative_reward: float
    task_score: float
    predicted_actions: List[str]
    expected_actions: List[str]
    predicted_focus: List[str]
    expected_focus: List[str]
    score_components: Dict[str, float]
    reward_totals: Dict[str, float]
