"""Environment contract tests for AgentValleyEnv.

These tests prove that the submission exposes the minimum OpenEnv-style loop:
reset -> observation, step(action) -> observation/reward/done/info, and terminal
machine-readable episode results.
"""

from __future__ import annotations

import unittest

from env.environment import ACTION_SPACE, OBSERVATION_SPACE, AgentValleyEnv
from env.schemas import Action


class TestEnvironmentContract(unittest.TestCase):
    def test_reset_returns_agent_safe_observation(self) -> None:
        env = AgentValleyEnv(difficulty="easy", seed=42)
        obs = env.reset()

        self.assertIsInstance(obs, dict)
        self.assertEqual(obs["difficulty"], "easy")
        self.assertIn("task_goal", obs)

        # Hidden verifier targets must never be visible to the policy.
        forbidden_keys = {
            "ground_truth_action",
            "acceptable_actions",
            "ground_truth_focus",
            "ground_truth_cooperation",
            "ground_truth_risk",
        }
        self.assertTrue(forbidden_keys.isdisjoint(obs.keys()))

    def test_step_returns_openenv_tuple_and_reward_breakdown(self) -> None:
        env = AgentValleyEnv(difficulty="easy", seed=42)
        env.reset()
        action = Action(
            primary_action="gather",
            focus_resource="food",
            cooperation_mode="share",
            risk_posture="balanced",
            rationale="Gather visible scarce food while sharing with the settlement.",
        )

        next_obs, reward, done, info = env.step(action)

        self.assertIsInstance(next_obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("reward_breakdown", info)
        self.assertIn("total", info["reward_breakdown"])
        self.assertEqual(info["reward_breakdown"]["total"], reward)

    def test_episode_terminates_with_machine_readable_result(self) -> None:
        env = AgentValleyEnv(difficulty="easy", seed=42)
        obs = env.reset()
        done = False
        info = {}

        while not done:
            # Safe generic baseline action. The exact score is not important here;
            # this test verifies terminal result structure.
            action = {
                "primary_action": "gather" if obs["food_supply"] < 0.6 else "cooperate",
                "focus_resource": "food" if obs["food_supply"] < 0.6 else "none",
                "cooperation_mode": "share" if obs["food_supply"] < 0.6 else "coordinate",
                "risk_posture": "balanced",
                "rationale": "Use only visible observation fields to choose a safe valley action.",
            }
            obs, _reward, done, info = env.step(action)

        result = info.get("episode_result")
        self.assertIsInstance(result, dict)
        self.assertIn(result["final_status"], {"valley_stabilized", "valley_survived", "valley_collapsed"})
        self.assertIn("task_score", result)
        self.assertIn("reward_totals", result)

    def test_declared_spaces_are_available(self) -> None:
        self.assertIn("primary_action", ACTION_SPACE["fields"])
        self.assertIn("food_supply", OBSERVATION_SPACE["fields"])


if __name__ == "__main__":
    unittest.main()
