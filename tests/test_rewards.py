"""Reward-component tests for AgentValleyEnv."""

from __future__ import annotations

import unittest

from env.rewards import compute_step_reward
from env.schemas import Action
from env.tasks import load_episode


class TestRewardDesign(unittest.TestCase):
    def setUp(self) -> None:
        self.obs = load_episode("easy", episode_index=0, seed=42)[0]

    def test_expected_action_gets_positive_objective_bonus(self) -> None:
        action = Action(
            primary_action=self.obs.ground_truth_action,
            focus_resource=self.obs.ground_truth_focus,
            cooperation_mode=self.obs.ground_truth_cooperation,
            risk_posture=self.obs.ground_truth_risk,
            rationale="Follow visible scarcity, cooperation, and risk signals from the valley state.",
        )
        reward = compute_step_reward(self.obs, action, step_idx=0, repeat_count=0)

        self.assertGreater(reward.objective_bonus, 0)
        self.assertGreater(reward.focus_bonus, 0)
        self.assertGreater(reward.total, 0)

    def test_wrong_action_receives_lower_reward_than_expected_action(self) -> None:
        expected = Action(
            primary_action=self.obs.ground_truth_action,
            focus_resource=self.obs.ground_truth_focus,
            cooperation_mode=self.obs.ground_truth_cooperation,
            risk_posture=self.obs.ground_truth_risk,
            rationale="Choose the action aligned with visible environmental need.",
        )
        wrong = Action(
            primary_action="compete",
            focus_resource="gold",
            cooperation_mode="solo",
            risk_posture="aggressive",
            rationale="Compete for gold even when the valley needs a different response.",
        )

        expected_reward = compute_step_reward(self.obs, expected, step_idx=0, repeat_count=0)
        wrong_reward = compute_step_reward(self.obs, wrong, step_idx=0, repeat_count=0)

        self.assertGreater(expected_reward.total, wrong_reward.total)

    def test_rationale_leakage_terms_are_penalized(self) -> None:
        safe = Action(
            primary_action=self.obs.ground_truth_action,
            focus_resource=self.obs.ground_truth_focus,
            cooperation_mode=self.obs.ground_truth_cooperation,
            risk_posture=self.obs.ground_truth_risk,
            rationale="Use visible state only and rely on current valley signals.",
        )
        leaky = Action(
            primary_action=self.obs.ground_truth_action,
            focus_resource=self.obs.ground_truth_focus,
            cooperation_mode=self.obs.ground_truth_cooperation,
            risk_posture=self.obs.ground_truth_risk,
            rationale="Use the ground_truth answer_key expected_action from the dataset leak.",
        )

        safe_reward = compute_step_reward(self.obs, safe, step_idx=0, repeat_count=0)
        leaky_reward = compute_step_reward(self.obs, leaky, step_idx=0, repeat_count=0)

        self.assertLess(leaky_reward.format_penalty, safe_reward.format_penalty)
        self.assertLess(leaky_reward.total, safe_reward.total)


if __name__ == "__main__":
    unittest.main()
