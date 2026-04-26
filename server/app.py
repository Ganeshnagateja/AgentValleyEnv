from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Environment imports
# ---------------------------------------------------------------------

try:
    from env.environment import AgentValleyEnv
except Exception as exc:
    raise RuntimeError(
        "Could not import AgentValleyEnv from env.environment. "
        "Make sure you are running from the project root."
    ) from exc


# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------

app = FastAPI(
    title="Agent Valley OpenEnv API",
    version="1.0.0",
    description="OpenEnv-style RL environment for Agent Valley.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "https://huggingface.co",
        "https://ganeshnagateja-agentvalleyenv.hf.space",
    ],
    allow_origin_regex=r"https://.*\.hf\.space",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: str = Field(default="easy")
    episode_index: int = Field(default=0)
    seed: Optional[int] = Field(default=42)


class StepRequest(BaseModel):
    action: Dict[str, Any]


ResetRequest.model_rebuild()
StepRequest.model_rebuild()


# ---------------------------------------------------------------------
# Global environment
# ---------------------------------------------------------------------

_current_env = AgentValleyEnv()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_state() -> Dict[str, Any]:
    try:
        return _current_env.state()
    except Exception:
        return {
            "status": "not_initialized",
            "message": "Environment state is not available. Call /reset first.",
        }


def _reset_env(
    difficulty: str = "easy",
    episode_index: int = 0,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    global _current_env

    _current_env = AgentValleyEnv(
        difficulty=difficulty,
        episode_index=episode_index,
        seed=seed,
    )

    observation = _current_env.reset()

    return {
        "observation": observation,
        "state": _current_env.state(),
        "done": False,
        "info": {
            "difficulty": difficulty,
            "episode_index": episode_index,
            "seed": seed,
            "message": "Environment reset successfully.",
        },
    }


# ---------------------------------------------------------------------
# API routes
# IMPORTANT:
# Do NOT use @app.get("/") for JSON.
# The root "/" is reserved for the React UI.
# ---------------------------------------------------------------------

@app.get("/api")
def api_root() -> Dict[str, Any]:
    return {
        "status": "AgentValleyEnv running",
        "environment": "AgentValleyEnv",
        "final_outputs": [
            "valley_stabilized",
            "valley_survived",
            "valley_collapsed",
        ],
        "tasks": {
            "easy": {
                "difficulty": "easy",
                "name": "Resource Stabilization",
                "max_steps": 5,
                "dataset": "easy.json",
                "description": "Resolve short-horizon resource and rest decisions before the settlement destabilizes.",
            },
            "medium": {
                "difficulty": "medium",
                "name": "Market Shock Coordination",
                "max_steps": 7,
                "dataset": "medium.json",
                "description": "Coordinate agents through market volatility, scarcity, and moderate external events.",
            },
            "hard": {
                "difficulty": "hard",
                "name": "Invasion Defense and Recovery",
                "max_steps": 10,
                "dataset": "hard.json",
                "description": "Balance defense, recovery, and resource routing during long-horizon valley crises.",
            },
        },
        "api_routes": [
            "/health",
            "/reset",
            "/step",
            "/state",
            "/action_space",
            "/observation_space",
            "/baseline",
            "/docs",
        ],
    }


@app.get("/health")
@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
@app.post("/api/reset")
def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    return _reset_env(
        difficulty=request.difficulty,
        episode_index=request.episode_index,
        seed=request.seed,
    )


@app.post("/step")
@app.post("/api/step")
def step(request: StepRequest) -> Dict[str, Any]:
    try:
        observation, reward, done, info = _current_env.step(request.action)

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }

    except RuntimeError as exc:
        if "Environment is done" in str(exc):
            current_state = _safe_state()
            return {
                "observation": current_state.get("observation", current_state),
                "reward": 0.0,
                "done": True,
                "info": {
                    "status": current_state.get("status", "episode_done"),
                    "message": "Episode already finished. Call /reset before next step.",
                    "requires_reset": True,
                },
            }

        raise


@app.get("/state")
@app.get("/api/state")
def state() -> Dict[str, Any]:
    return _safe_state()


@app.get("/action_space")
@app.get("/api/action_space")
def action_space() -> Dict[str, Any]:
    return {
        "primary_action": [
            "gather",
            "trade",
            "build",
            "explore",
            "rest",
            "cooperate",
            "compete",
            "defend",
        ],
        "focus_resource": ["food", "wood", "stone", "gold", "ore"],
        "cooperation_mode": ["solo", "share", "coordinate"],
        "risk_posture": ["cautious", "balanced", "bold"],
        "required_fields": [
            "primary_action",
            "focus_resource",
            "cooperation_mode",
            "risk_posture",
            "rationale",
        ],
    }


@app.get("/observation_space")
@app.get("/api/observation_space")
def observation_space() -> Dict[str, Any]:
    return {
        "fields": [
            "tick",
            "difficulty",
            "scenario",
            "task_goal",
            "region",
            "active_agents",
            "food_supply",
            "wood_supply",
            "stone_supply",
            "gold_supply",
            "ore_supply",
            "average_health",
            "average_energy",
            "cooperation_index",
            "threat_level",
            "market_volatility",
            "event_severity",
            "region_danger",
            "defense_readiness",
        ],
        "note": "Hidden ground-truth fields are removed before observations are returned.",
    }


@app.get("/baseline")
@app.get("/api/baseline")
def baseline() -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for difficulty in ["easy", "medium", "hard"]:
        env = AgentValleyEnv(difficulty=difficulty, episode_index=0, seed=42)
        env.reset()

        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < 20:
            action = {
                "primary_action": "gather",
                "focus_resource": "food",
                "cooperation_mode": "share",
                "risk_posture": "balanced",
                "rationale": "baseline policy selects a safe cooperative resource action",
            }

            _, reward, done, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

        final_state = env.state()
        results[difficulty] = {
            "total_reward": round(total_reward, 4),
            "steps": steps,
            "status": final_state.get("status", "unknown"),
            "done": done,
        }

    return {
        "results": results,
        "final_output_labels": [
            "valley_stabilized",
            "valley_survived",
            "valley_collapsed",
        ],
    }


# ---------------------------------------------------------------------
# React frontend serving
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT_DIR / "app" / "dist"

if (FRONTEND_DIST / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="frontend-assets",
    )


@app.get("/", include_in_schema=False)
def serve_frontend():
    index_file = FRONTEND_DIST / "index.html"

    if index_file.exists():
        return FileResponse(str(index_file))

    return {
        "status": "AgentValleyEnv running",
        "message": "Frontend build not found. Run: cd app && npm install && npm run build",
        "api": "/api",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend_routes(full_path: str):
    api_prefixes = (
        "api",
        "health",
        "reset",
        "step",
        "state",
        "action_space",
        "observation_space",
        "baseline",
        "docs",
        "openapi.json",
    )

    if full_path.startswith(api_prefixes):
        return {"error": "API route not found"}

    requested_file = FRONTEND_DIST / full_path
    if requested_file.exists() and requested_file.is_file():
        return FileResponse(str(requested_file))

    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    return {
        "status": "AgentValleyEnv running",
        "message": "Frontend build not found. Run: cd app && npm install && npm run build",
        "api": "/api",
        "health": "/health",
        "docs": "/docs",
    }


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
