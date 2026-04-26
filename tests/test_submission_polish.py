from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from env.environment import AgentValleyEnv
from env.openenv_compat import Environment


ROOT = Path(__file__).resolve().parent.parent


class TestSubmissionPolish(unittest.TestCase):
    def test_plot_generation_creates_assets(self) -> None:
        subprocess.run([sys.executable, "scripts/generate_training_plots.py"], cwd=ROOT, check=True)
        self.assertTrue((ROOT / "assets" / "reward_curve.png").exists())
        self.assertTrue((ROOT / "assets" / "loss_curve.png").exists())
        summary_path = ROOT / "assets" / "training_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("generated_from", summary)
        self.assertIn("best_reward", summary)

    def test_readme_contains_evidence_and_deliverables(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("assets/reward_curve.png", readme)
        self.assertIn("assets/loss_curve.png", readme)
        self.assertIn("## Deliverables", readme)
        self.assertIn("Hugging Face Space: TODO", readme)
        self.assertIn("notebooks/agent_valley_training.ipynb", readme)
        self.assertIn("notebooks/multi_agent_training.ipynb", readme)
        self.assertIn("docs/writeup.md", readme)

    def test_agent_valley_env_inherits_openenv_compat_base(self) -> None:
        self.assertTrue(issubclass(AgentValleyEnv, Environment))
        self.assertIsInstance(AgentValleyEnv(), Environment)

    def test_artifact_metric_paths_are_relative(self) -> None:
        subprocess.run([sys.executable, "scripts/generate_training_plots.py"], cwd=ROOT, check=True)
        paths = [
            ROOT / "artifacts" / "q_learning" / "metrics.jsonl",
            ROOT / "artifacts" / "neural_policy" / "metrics.jsonl",
            ROOT / "artifacts" / "grpo" / "metrics.jsonl",
            ROOT / "artifacts" / "multi_agent_grpo" / "metrics.jsonl",
            ROOT / "assets" / "training_summary.json",
        ]
        forbidden = ("C:\\\\Users", "C:/Users", "VADDE", "OneDrive", "\\u30c9", "ドキュメント")
        for path in paths:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path} contains local path token {token}")


if __name__ == "__main__":
    unittest.main()
