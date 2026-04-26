"""FastAPI wrapper for AgentValleyEnv.

Routes are available at BOTH:
  /reset  /step  /state  ...           (OpenEnv spec – external trainers)
  /api/reset  /api/step  /api/state    (frontend – React UI)

The built React app (app/dist) is served as static files so that a single
Docker container on Hugging Face Spaces serves the full demo.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from baseline_eval import RuleBasedAgent, run_episode
from env.action_space import action_count, list_all_actions
from env.agents import AGENT_IDS
from env.environment import AgentValleyEnv
from env.multi_agent_env import MultiAgentValleyEnv
from env.tasks import list_tasks
from training.manager import training_manager
from training.policy_runtime import evaluate_policy, policy_action


class ResetRequest(BaseModel):
    difficulty: str = "easy"
    episode_index: int = 0
    seed: int = 42


class StepRequest(BaseModel):
    action: Dict[str, Any]


class TrainingStartRequest(BaseModel):
    mode: str = "grpo"
    episodes: int = 10
    difficulty: str = "mixed"
    seed: int = 42
    reset_metrics: bool = True
    alpha: float = 0.25
    gamma: float = 0.95
    epsilon: float = 0.35
    learning_rate: float = 0.002
    group_size: int = 4


class PolicyActionRequest(BaseModel):
    mode: str = "grpo"
    observation: Dict[str, Any]
    deterministic: bool = True
    seed: int = 42


class PolicyEvaluateRequest(BaseModel):
    mode: str = "grpo"
    difficulty: str = "easy"
    episodes: int = 3
    seed: int = 42
    deterministic: bool = True


class MultiAgentStepRequest(BaseModel):
    actions: Dict[str, Dict[str, Any]]


class MultiAgentEvaluateRequest(BaseModel):
    difficulty: str = "hard"
    episodes: int = 3
    seed: int = 42
    coordinated: bool = True


app = FastAPI(title="AgentValleyEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_current_env: Optional[AgentValleyEnv] = None
_current_multi_agent_env: Optional[MultiAgentValleyEnv] = None


# ── Shared logic ────────────────────────────────────────────────────────────

def _do_home():
    return {
        "status": "AgentValleyEnv running",
        "final_outputs": ["valley_stabilized", "valley_survived", "valley_collapsed"],
        "tasks": list_tasks(),
    }

def _do_health():
    return {"status": "ok"}

def _do_reset(req: ResetRequest):
    global _current_env
    _current_env = AgentValleyEnv(
        difficulty=req.difficulty,
        episode_index=req.episode_index,
        seed=req.seed,
    )
    observation = _current_env.reset()
    return {"observation": observation, "state": _current_env.state()}

def _do_step(req: StepRequest):
    global _current_env
    if _current_env is None:
        _current_env = AgentValleyEnv()
        _current_env.reset()
    try:
        observation, reward, done, info = _current_env.step(req.action)
    except RuntimeError as exc:
        if str(exc) == "Environment is done. Call reset() before step().":
            return {
                "observation": None,
                "reward": 0.0,
                "done": True,
                "info": {
                    "requires_reset": True,
                    "message": "Episode already finished. Call /reset before next step.",
                },
            }
        raise
    return {"observation": observation, "reward": reward, "done": done, "info": info}

def _do_state():
    if _current_env is None:
        return {"status": "not_reset"}
    return _current_env.state()

def _do_action_space():
    return {
        **AgentValleyEnv.action_space,
        "discrete_action_count": action_count(),
        "discrete_actions": list_all_actions(),
    }

def _do_observation_space():
    return AgentValleyEnv.observation_space

def _do_baseline(difficulty, episode_index, seed):
    return run_episode(RuleBasedAgent(), difficulty, episode_index, seed)


def _do_training_start(request: TrainingStartRequest):
    return training_manager.start(request.model_dump(mode="json"))


def _do_training_stop():
    return training_manager.stop()


def _do_training_status():
    return training_manager.status()


def _do_training_metrics(mode: Optional[str] = None, limit: int = 200):
    return training_manager.metrics(mode=mode, limit=limit)


def _do_training_latest(mode: Optional[str] = None):
    return training_manager.latest(mode=mode)


def _do_policy_action(request: PolicyActionRequest):
    return policy_action(
        mode=request.mode,
        observation=request.observation,
        deterministic=request.deterministic,
        seed=request.seed,
    )


def _do_policy_evaluate(request: PolicyEvaluateRequest):
    return evaluate_policy(
        mode=request.mode,
        difficulty=request.difficulty,
        episodes=request.episodes,
        seed=request.seed,
        deterministic=request.deterministic,
    )


def _coordinated_multi_agent_actions() -> Dict[str, Dict[str, Any]]:
    return {
        "farmer": {
            "primary_action": "gather",
            "focus_resource": "food",
            "cooperation_mode": "share",
            "risk_posture": "balanced",
            "rationale": "Food production supports the team while others specialize.",
        },
        "miner": {
            "primary_action": "gather",
            "focus_resource": "ore",
            "cooperation_mode": "coordinate",
            "risk_posture": "balanced",
            "rationale": "Ore supports coordinated construction and recovery.",
        },
        "builder": {
            "primary_action": "build",
            "focus_resource": "stone",
            "cooperation_mode": "coordinate",
            "risk_posture": "cautious",
            "rationale": "Building with stone improves shared infrastructure.",
        },
        "warrior": {
            "primary_action": "defend",
            "focus_resource": "none",
            "cooperation_mode": "protect",
            "risk_posture": "cautious",
            "rationale": "Defense protects the team during risky valley conditions.",
        },
    }


def _selfish_multi_agent_actions() -> Dict[str, Dict[str, Any]]:
    return {
        agent_id: {
            "primary_action": "gather",
            "focus_resource": "food",
            "cooperation_mode": "solo",
            "risk_posture": "aggressive",
            "rationale": "This baseline piles on one resource without team coordination.",
        }
        for agent_id in AGENT_IDS
    }


def _do_multi_agent_reset(req: ResetRequest):
    global _current_multi_agent_env
    _current_multi_agent_env = MultiAgentValleyEnv(
        difficulty=req.difficulty,
        episode_index=req.episode_index,
        seed=req.seed,
    )
    observations = _current_multi_agent_env.reset(episode_index=req.episode_index, seed=req.seed)
    return {"observations": observations, "state": _current_multi_agent_env.state()}


def _do_multi_agent_state():
    if _current_multi_agent_env is None:
        return {"status": "not_reset"}
    return _current_multi_agent_env.state()


def _do_multi_agent_step(req: MultiAgentStepRequest):
    global _current_multi_agent_env
    if _current_multi_agent_env is None:
        _current_multi_agent_env = MultiAgentValleyEnv()
        _current_multi_agent_env.reset()
    try:
        observations, rewards, dones, info = _current_multi_agent_env.step(req.actions)
    except RuntimeError as exc:
        if str(exc) == "Environment is done. Call reset() before step().":
            return {
                "observations": None,
                "rewards": {agent_id: 0.0 for agent_id in AGENT_IDS},
                "dones": {**{agent_id: True for agent_id in AGENT_IDS}, "__all__": True},
                "info": {
                    "requires_reset": True,
                    "message": "Episode already finished. Call /api/multi-agent/reset before next step.",
                },
            }
        raise
    return {"observations": observations, "rewards": rewards, "dones": dones, "info": info}


def _do_multi_agent_evaluate(req: MultiAgentEvaluateRequest):
    rows = []
    action_factory = _coordinated_multi_agent_actions if req.coordinated else _selfish_multi_agent_actions
    for episode in range(1, req.episodes + 1):
        env = MultiAgentValleyEnv(difficulty=req.difficulty, episode_index=episode - 1, seed=req.seed)
        env.reset(episode_index=episode - 1, seed=req.seed)
        done = False
        total_team_reward = 0.0
        cooperation_steps = 0
        steps = 0
        info: Dict[str, Any] = {}
        while not done:
            _obs, rewards, dones, info = env.step(action_factory())
            total_team_reward += sum(float(rewards[agent_id]) for agent_id in AGENT_IDS)
            cooperation_steps += 1 if float(info.get("cooperation_bonus", 0.0)) > 0 else 0
            steps += 1
            done = bool(dones.get("__all__"))
        episode_result = info.get("episode_result") or {}
        rows.append(
            {
                "episode": episode,
                "difficulty": req.difficulty,
                "total_team_reward": round(total_team_reward, 4),
                "cooperation_rate": round(cooperation_steps / max(steps, 1), 4),
                "task_score": float(episode_result.get("task_score", 0.0)),
                "final_status": episode_result.get("final_status", "unknown"),
                "steps": steps,
            }
        )
    avg_reward = sum(row["total_team_reward"] for row in rows) / max(len(rows), 1)
    return {
        "mode": "multi_agent_eval",
        "coordinated": req.coordinated,
        "average_total_team_reward": round(avg_reward, 4),
        "results": rows,
    }


# ── Root-level routes (OpenEnv spec) ───────────────────────────────────────

@app.get("/")
def home(): return _do_home()

@app.get("/health")
def health(): return _do_health()

@app.post("/reset")
def reset(request: Optional[ResetRequest] = None): return _do_reset(request or ResetRequest())

@app.post("/step")
def step(request: StepRequest): return _do_step(request)

@app.get("/state")
def state(): return _do_state()

@app.get("/action_space")
def action_space(): return _do_action_space()

@app.get("/observation_space")
def observation_space(): return _do_observation_space()

@app.get("/baseline")
def baseline(difficulty: str = "easy", episode_index: int = 0, seed: int = 42):
    return _do_baseline(difficulty, episode_index, seed)

@app.get("/training/modes")
def training_modes(): return training_manager.modes()

@app.post("/training/start")
def training_start(request: TrainingStartRequest): return _do_training_start(request)

@app.post("/training/stop")
def training_stop(): return _do_training_stop()

@app.get("/training/status")
def training_status(): return _do_training_status()

@app.get("/training/metrics")
def training_metrics(mode: Optional[str] = None, limit: int = 200): return _do_training_metrics(mode, limit)

@app.get("/training/latest")
def training_latest(mode: Optional[str] = None): return _do_training_latest(mode)

@app.post("/policy/action")
def policy_action_route(request: PolicyActionRequest): return _do_policy_action(request)

@app.post("/policy/evaluate")
def policy_evaluate_route(request: PolicyEvaluateRequest): return _do_policy_evaluate(request)

@app.post("/multi-agent/reset")
def multi_agent_reset(request: Optional[ResetRequest] = None): return _do_multi_agent_reset(request or ResetRequest(difficulty="hard"))

@app.get("/multi-agent/state")
def multi_agent_state(): return _do_multi_agent_state()

@app.post("/multi-agent/step")
def multi_agent_step(request: MultiAgentStepRequest): return _do_multi_agent_step(request)

@app.post("/multi-agent/evaluate")
def multi_agent_evaluate(request: MultiAgentEvaluateRequest): return _do_multi_agent_evaluate(request)


# ── /api routes (React frontend) ───────────────────────────────────────────

@app.get("/api")
def api_home(): return _do_home()

@app.get("/api/health")
def api_health(): return _do_health()

@app.post("/api/reset")
def api_reset(request: Optional[ResetRequest] = None): return _do_reset(request or ResetRequest())

@app.post("/api/step")
def api_step(request: StepRequest): return _do_step(request)

@app.get("/api/state")
def api_state(): return _do_state()

@app.get("/api/action_space")
def api_action_space(): return _do_action_space()

@app.get("/api/observation_space")
def api_observation_space(): return _do_observation_space()

@app.get("/api/baseline")
def api_baseline(difficulty: str = "easy", episode_index: int = 0, seed: int = 42):
    return _do_baseline(difficulty, episode_index, seed)

@app.get("/api/training/modes")
def api_training_modes(): return training_manager.modes()

@app.post("/api/training/start")
def api_training_start(request: TrainingStartRequest): return _do_training_start(request)

@app.post("/api/training/stop")
def api_training_stop(): return _do_training_stop()

@app.get("/api/training/status")
def api_training_status(): return _do_training_status()

@app.get("/api/training/metrics")
def api_training_metrics(mode: Optional[str] = None, limit: int = 200): return _do_training_metrics(mode, limit)

@app.get("/api/training/latest")
def api_training_latest(mode: Optional[str] = None): return _do_training_latest(mode)

@app.post("/api/policy/action")
def api_policy_action(request: PolicyActionRequest): return _do_policy_action(request)

@app.post("/api/policy/evaluate")
def api_policy_evaluate(request: PolicyEvaluateRequest): return _do_policy_evaluate(request)

@app.post("/api/multi-agent/reset")
def api_multi_agent_reset(request: Optional[ResetRequest] = None): return _do_multi_agent_reset(request or ResetRequest(difficulty="hard"))

@app.get("/api/multi-agent/state")
def api_multi_agent_state(): return _do_multi_agent_state()

@app.post("/api/multi-agent/step")
def api_multi_agent_step(request: MultiAgentStepRequest): return _do_multi_agent_step(request)

@app.post("/api/multi-agent/evaluate")
def api_multi_agent_evaluate(request: MultiAgentEvaluateRequest): return _do_multi_agent_evaluate(request)


# ── Static file serving (React build at app/dist) ──────────────────────────

_DIST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "dist")

if os.path.isdir(os.path.join(_DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(_DIST, "assets")), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    candidate = os.path.join(_DIST, full_path)
    if os.path.isfile(candidate):
        return FileResponse(candidate)
    index = os.path.join(_DIST, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return {"status": "AgentValleyEnv running"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
