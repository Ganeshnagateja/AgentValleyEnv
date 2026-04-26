from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from training.neural_policy import load_checkpoint
from training.train_neural_policy import NeuralPolicyConfig, NeuralPolicyTrainer


class TestNeuralPolicy(unittest.TestCase):
    def test_parameters_change_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = NeuralPolicyTrainer(
                NeuralPolicyConfig(episodes=1, difficulty="easy", seed=11, artifact_dir=Path(tmp), reset_metrics=True)
            )
            before = [param.detach().clone() for param in trainer.policy.parameters()]
            metrics = trainer.train()
            after = [param.detach().clone() for param in trainer.policy.parameters()]
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
            self.assertTrue(trainer.checkpoint_path.exists())
            self.assertTrue(trainer.metrics_path.exists())
            loaded, metadata = load_checkpoint(trainer.checkpoint_path)
            self.assertTrue(metadata["checkpoint_found"])
            self.assertEqual(len(list(loaded.parameters())), len(list(trainer.policy.parameters())))
            self.assertIn("policy_loss", metrics[0])


if __name__ == "__main__":
    unittest.main()
