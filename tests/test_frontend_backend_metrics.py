from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class TestFrontendBackendMetrics(unittest.TestCase):
    def test_training_dashboard_uses_backend_metric_fields(self) -> None:
        dashboard = (ROOT / "app" / "src" / "sections" / "TrainingDashboard.tsx").read_text(encoding="utf-8")
        api_client = (ROOT / "app" / "src" / "lib" / "apiClient.ts").read_text(encoding="utf-8")

        self.assertIn("trainingMetrics", api_client)
        self.assertIn("/training/metrics", api_client)
        self.assertIn("trainingStatus", api_client)
        self.assertIn("/training/status", api_client)
        self.assertIn("trainingLatest", api_client)
        self.assertIn("/training/latest", api_client)
        self.assertIn("multi_agent_grpo", dashboard)
        self.assertIn("multi_agent_grpo", api_client)
        for field in (
            "total_reward",
            "total_team_reward",
            "cooperation_rate",
            "policy_loss",
            "mean_q_value",
            "max_q_value",
            "kl_divergence",
            "entropy",
        ):
            self.assertIn(field, dashboard)

    def test_no_fake_training_curve_randomness(self) -> None:
        files = [
            ROOT / "app" / "src" / "sections" / "TrainingDashboard.tsx",
            ROOT / "app" / "src" / "game" / "TrainingEngine.ts",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
        self.assertNotIn("Math.random", combined)
        self.assertNotIn("simulateGRPOUpdate", combined)
        self.assertNotIn("Simulate loss", combined)
        self.assertNotIn("successfulActions.map", combined)


if __name__ == "__main__":
    unittest.main()
