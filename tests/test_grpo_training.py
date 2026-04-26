from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from training.grpo_train import GRPOConfig, GRPOTrainer


class TestGRPOTraining(unittest.TestCase):
    def test_parameters_change_and_metrics_are_real(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = GRPOTrainer(
                GRPOConfig(
                    episodes=1,
                    difficulty="easy",
                    seed=17,
                    group_size=3,
                    artifact_dir=Path(tmp),
                    reset_metrics=True,
                )
            )
            before = [param.detach().clone() for param in trainer.policy.parameters()]
            metrics = trainer.train()
            after = [param.detach().clone() for param in trainer.policy.parameters()]
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
            self.assertTrue(trainer.checkpoint_path.exists())
            self.assertTrue(trainer.metrics_path.exists())
            metric = metrics[0]
            self.assertIn("policy_loss", metric)
            self.assertIn("kl_divergence", metric)
            self.assertIn("entropy", metric)
            self.assertIn("mean_group_reward", metric)

    def test_policy_loss_uses_log_probs_and_advantages(self) -> None:
        source = inspect.getsource(GRPOTrainer._update_from_group)
        self.assertIn("old_log_probs", source)
        self.assertIn("advantages", source)
        self.assertIn("torch.minimum", source)
        self.assertIn("self.optimizer.step()", source)
        self.assertNotIn("random()", source)


if __name__ == "__main__":
    unittest.main()
