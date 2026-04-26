"""Thread-safe training job manager for the FastAPI backend."""

from __future__ import annotations

import threading
from typing import Any

from training.common import ARTIFACT_ROOT, project_relative, read_jsonl, resolve_project_path, utc_now
from training.grpo_train import GRPOConfig, GRPOTrainer
from training.ma_grpo_train import MAGRPOConfig, MAGRPOTrainer
from training.q_learning import QLearningConfig, QLearningTrainer
from training.train_neural_policy import NeuralPolicyConfig, NeuralPolicyTrainer


TRAINING_MODES = {
    "q_learning": {
        "label": "Tabular Q-learning",
        "metrics_path": project_relative(ARTIFACT_ROOT / "q_learning" / "metrics.jsonl"),
        "checkpoint_path": project_relative(ARTIFACT_ROOT / "q_learning" / "q_table.json"),
    },
    "neural_policy": {
        "label": "Neural policy gradient",
        "metrics_path": project_relative(ARTIFACT_ROOT / "neural_policy" / "metrics.jsonl"),
        "checkpoint_path": project_relative(ARTIFACT_ROOT / "neural_policy" / "policy.pt"),
    },
    "grpo": {
        "label": "GRPO-style clipped policy optimization",
        "metrics_path": project_relative(ARTIFACT_ROOT / "grpo" / "metrics.jsonl"),
        "checkpoint_path": project_relative(ARTIFACT_ROOT / "grpo" / "policy.pt"),
    },
    "multi_agent_grpo": {
        "label": "Multi-agent GRPO-style role policies",
        "metrics_path": project_relative(ARTIFACT_ROOT / "multi_agent_grpo" / "metrics.jsonl"),
        "checkpoint_path": project_relative(ARTIFACT_ROOT / "multi_agent_grpo"),
    },
}


class TrainingManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status: dict[str, Any] = {
            "running": False,
            "mode": None,
            "episode": 0,
            "episodes_requested": 0,
            "difficulty": None,
            "seed": None,
            "latest_metric": None,
            "error": None,
            "started_at": None,
            "updated_at": None,
            "checkpoint_path": None,
            "metrics_path": None,
        }

    def modes(self) -> dict[str, Any]:
        return {"modes": TRAINING_MODES}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _set_status(self, **updates: Any) -> None:
        with self._lock:
            self._status.update(updates)
            self._status["updated_at"] = utc_now()

    def _on_metric(self, metric: dict[str, Any]) -> None:
        self._set_status(
            episode=metric.get("episode"),
            difficulty=metric.get("difficulty"),
            latest_metric=metric,
            checkpoint_path=metric.get("checkpoint_path"),
            metrics_path=metric.get("metrics_path"),
        )

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("mode", "grpo")
        if mode not in TRAINING_MODES:
            raise ValueError(f"Unknown training mode '{mode}'")
        with self._lock:
            if self._status.get("running"):
                raise RuntimeError("A training job is already running")

        episodes = int(payload.get("episodes", 10))
        difficulty = str(payload.get("difficulty", "mixed"))
        seed = int(payload.get("seed", 42))
        reset_metrics = bool(payload.get("reset_metrics", True))
        self._stop_event = threading.Event()

        if mode == "q_learning":
            trainer = QLearningTrainer(
                QLearningConfig(
                    episodes=episodes,
                    difficulty=difficulty,
                    seed=seed,
                    alpha=float(payload.get("alpha", 0.25)),
                    gamma=float(payload.get("gamma", 0.95)),
                    epsilon=float(payload.get("epsilon", 0.35)),
                    reset_metrics=reset_metrics,
                )
            )
        elif mode == "neural_policy":
            trainer = NeuralPolicyTrainer(
                NeuralPolicyConfig(
                    episodes=episodes,
                    difficulty=difficulty,
                    seed=seed,
                    learning_rate=float(payload.get("learning_rate", 0.002)),
                    reset_metrics=reset_metrics,
                )
            )
        elif mode == "grpo":
            trainer = GRPOTrainer(
                GRPOConfig(
                    episodes=episodes,
                    difficulty=difficulty,
                    seed=seed,
                    group_size=int(payload.get("group_size", 4)),
                    learning_rate=float(payload.get("learning_rate", 0.0015)),
                    reset_metrics=reset_metrics,
                )
            )
        else:
            trainer = MAGRPOTrainer(
                MAGRPOConfig(
                    episodes=episodes,
                    difficulty=difficulty,
                    seed=seed,
                    group_size=int(payload.get("group_size", 4)),
                    learning_rate=float(payload.get("learning_rate", 0.0015)),
                    reset_metrics=reset_metrics,
                )
            )

        self._set_status(
            running=True,
            mode=mode,
            episode=0,
            episodes_requested=episodes,
            difficulty=difficulty,
            seed=seed,
            latest_metric=None,
            error=None,
            started_at=utc_now(),
            checkpoint_path=TRAINING_MODES[mode]["checkpoint_path"],
            metrics_path=TRAINING_MODES[mode]["metrics_path"],
        )

        def run() -> None:
            try:
                trainer.train(stop_event=self._stop_event, metric_callback=self._on_metric)
                stopped = self._stop_event.is_set()
                self._set_status(running=False, stopped=stopped, completed=not stopped)
            except Exception as exc:  # pragma: no cover - surfaced through API
                self._set_status(running=False, error=str(exc), completed=False)

        self._thread = threading.Thread(target=run, name=f"agent-valley-{mode}", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        self._set_status(stopped=True)
        return self.status()

    def metrics(self, mode: str | None = None, limit: int = 200) -> dict[str, Any]:
        status = self.status()
        selected_mode = mode or status.get("mode") or "grpo"
        if selected_mode not in TRAINING_MODES:
            raise ValueError(f"Unknown training mode '{selected_mode}'")
        path = resolve_project_path(TRAINING_MODES[selected_mode]["metrics_path"])
        return {
            "mode": selected_mode,
            "metrics": read_jsonl(path, limit=limit),
            "metrics_path": TRAINING_MODES[selected_mode]["metrics_path"],
            "updated_at": utc_now(),
        }

    def latest(self, mode: str | None = None) -> dict[str, Any]:
        payload = self.metrics(mode=mode, limit=1)
        latest = payload["metrics"][-1] if payload["metrics"] else None
        return {
            "mode": payload["mode"],
            "latest_metric": latest,
            "metrics_path": payload["metrics_path"],
            "updated_at": utc_now(),
        }


training_manager = TrainingManager()
