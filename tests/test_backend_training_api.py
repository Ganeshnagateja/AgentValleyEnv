from __future__ import annotations

import time
import unittest

from fastapi.testclient import TestClient

from server.app import app


class TestBackendTrainingAPI(unittest.TestCase):
    def test_step_after_done_returns_requires_reset_json(self) -> None:
        client = TestClient(app)
        client.post("/reset", json={"difficulty": "easy", "episode_index": 0, "seed": 42})
        action = {
            "primary_action": "gather",
            "focus_resource": "food",
            "cooperation_mode": "share",
            "risk_posture": "balanced",
            "rationale": "Use visible valley signals for a safe test action.",
        }
        done = False
        response = None
        while not done:
            response = client.post("/step", json={"action": action})
            self.assertEqual(response.status_code, 200)
            done = response.json()["done"]

        response = client.post("/step", json={"action": action})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["observation"])
        self.assertEqual(payload["reward"], 0.0)
        self.assertTrue(payload["done"])
        self.assertTrue(payload["info"]["requires_reset"])

    def test_training_modes_start_metrics_and_evaluate(self) -> None:
        client = TestClient(app)
        modes = client.get("/api/training/modes")
        self.assertEqual(modes.status_code, 200)
        self.assertIn("q_learning", modes.json()["modes"])
        self.assertIn("multi_agent_grpo", modes.json()["modes"])

        started = client.post(
            "/api/training/start",
            json={"mode": "q_learning", "episodes": 1, "difficulty": "easy", "seed": 101, "reset_metrics": True},
        )
        self.assertEqual(started.status_code, 200)

        status_payload = {}
        for _ in range(50):
            status_payload = client.get("/api/training/status").json()
            if not status_payload.get("running"):
                break
            time.sleep(0.1)
        self.assertFalse(status_payload.get("running"))
        self.assertIsNone(status_payload.get("error"))

        metrics = client.get("/api/training/metrics?mode=q_learning&limit=5")
        self.assertEqual(metrics.status_code, 200)
        rows = metrics.json()["metrics"]
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("total_reward", rows[-1])
        self.assertIn("mean_q_value", rows[-1])

        evaluation = client.post(
            "/api/policy/evaluate",
            json={"mode": "q_learning", "difficulty": "easy", "episodes": 1, "seed": 101},
        )
        self.assertEqual(evaluation.status_code, 200)
        result = evaluation.json()
        self.assertEqual(result["mode"], "q_learning")
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("task_score", result["results"][0])

    def test_multi_agent_training_mode_start_and_metrics(self) -> None:
        client = TestClient(app)
        started = client.post(
            "/api/training/start",
            json={"mode": "multi_agent_grpo", "episodes": 1, "difficulty": "easy", "seed": 202, "group_size": 2, "reset_metrics": True},
        )
        self.assertEqual(started.status_code, 200)

        status_payload = {}
        for _ in range(80):
            status_payload = client.get("/api/training/status").json()
            if not status_payload.get("running"):
                break
            time.sleep(0.1)
        self.assertFalse(status_payload.get("running"))
        self.assertIsNone(status_payload.get("error"))

        metrics = client.get("/api/training/metrics?mode=multi_agent_grpo&limit=5")
        self.assertEqual(metrics.status_code, 200)
        rows = metrics.json()["metrics"]
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[-1]["mode"], "multi_agent_grpo")
        self.assertIn("total_team_reward", rows[-1])
        self.assertIn("cooperation_rate", rows[-1])

    def test_multi_agent_api_reset_step_and_evaluate(self) -> None:
        client = TestClient(app)
        reset = client.post("/api/multi-agent/reset", json={"difficulty": "easy", "episode_index": 0, "seed": 42})
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(set(reset.json()["observations"]), {"farmer", "miner", "builder", "warrior"})

        actions = {
            "farmer": {"primary_action": "gather", "focus_resource": "food", "cooperation_mode": "share", "risk_posture": "balanced", "rationale": "Food supports the shared plan."},
            "miner": {"primary_action": "gather", "focus_resource": "ore", "cooperation_mode": "coordinate", "risk_posture": "balanced", "rationale": "Ore supports the shared plan."},
            "builder": {"primary_action": "build", "focus_resource": "stone", "cooperation_mode": "coordinate", "risk_posture": "cautious", "rationale": "Stone building supports the shared plan."},
            "warrior": {"primary_action": "defend", "focus_resource": "none", "cooperation_mode": "protect", "risk_posture": "cautious", "rationale": "Defense protects the shared plan."},
        }
        step = client.post("/api/multi-agent/step", json={"actions": actions})
        self.assertEqual(step.status_code, 200)
        payload = step.json()
        self.assertEqual(set(payload["rewards"]), {"farmer", "miner", "builder", "warrior"})
        self.assertIn("cooperation_bonus", payload["info"])

        evaluation = client.post("/api/multi-agent/evaluate", json={"difficulty": "easy", "episodes": 1, "seed": 42, "coordinated": True})
        self.assertEqual(evaluation.status_code, 200)
        self.assertIn("average_total_team_reward", evaluation.json())


if __name__ == "__main__":
    unittest.main()
