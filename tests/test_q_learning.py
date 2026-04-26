from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from env.action_space import action_count
from training.q_learning import QLearningConfig, QLearningTrainer


class TestQLearning(unittest.TestCase):
    def test_bellman_update_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = QLearningTrainer(
                QLearningConfig(alpha=0.5, gamma=0.9, artifact_dir=Path(tmp), reset_metrics=True)
            )
            trainer.q_table["next"] = [0.0 for _ in range(action_count())]
            trainer.q_table["next"][3] = 2.0
            updated = trainer.bellman_update("state", 0, reward=1.0, next_state_key="next", done=False)
            self.assertAlmostEqual(updated, 0.5 * (1.0 + 0.9 * 2.0), places=6)

    def test_training_changes_q_table_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = QLearningTrainer(
                QLearningConfig(episodes=1, difficulty="easy", seed=7, artifact_dir=Path(tmp), reset_metrics=True)
            )
            metrics = trainer.train()
            self.assertEqual(len(metrics), 1)
            self.assertTrue(trainer.q_table_path.exists())
            self.assertTrue(trainer.metrics_path.exists())
            payload = json.loads(trainer.q_table_path.read_text(encoding="utf-8"))
            values = [value for row in payload["q_table"].values() for value in row]
            self.assertTrue(any(value != 0.0 for value in values))
            self.assertIn("total_reward", metrics[0])
            self.assertIn("mean_q_value", metrics[0])


if __name__ == "__main__":
    unittest.main()
