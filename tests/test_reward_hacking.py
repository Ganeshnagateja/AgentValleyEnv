"""Anti-reward-hacking tests for AgentValleyEnv.

These tests are intentionally adversarial. They check that hidden labels are not
visible, invalid schemas are rejected, and repeated action farming is penalized.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from env.environment import AgentValleyEnv
from env.schemas import Action


class TestRewardHackingResistance(unittest.TestCase):
    def test_hidden_answer_keys_are_not_in_observation(self) -> None:
        env = AgentValleyEnv(difficulty="medium", seed=7)
        obs = env.reset()

        hidden_keys = [key for key in obs if "ground_truth" in key or key == "acceptable_actions"]
        self.assertEqual(hidden_keys, [])

    def test_extra_action_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Action(
                primary_action="gather",
                focus_resource="food",
                cooperation_mode="share",
                risk_posture="balanced",
                rationale="Try to smuggle reward control through an extra field.",
                set_reward=999,  # type: ignore[call-arg]
            )

    def test_invalid_action_enum_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Action(
                primary_action="edit_reward",
                focus_resource="food",
                cooperation_mode="share",
                risk_posture="balanced",
                rationale="Attempt to directly edit the reward instead of acting.",
            )

    def test_repeated_action_farming_triggers_loop_penalty(self) -> None:
        env = AgentValleyEnv(difficulty="hard", seed=42)
        env.reset()
        action = {
            "primary_action": "gather",
            "focus_resource": "food",
            "cooperation_mode": "share",
            "risk_posture": "balanced",
            "rationale": "Repeat the same visible action to test loop penalties.",
        }

        loop_penalties = []
        done = False
        while not done and len(loop_penalties) < 4:
            _obs, _reward, done, info = env.step(action)
            loop_penalties.append(info["reward_breakdown"]["loop_penalty"])

        self.assertEqual(loop_penalties[0], 0.0)
        self.assertLessEqual(loop_penalties[1], 0.0)
        self.assertLess(loop_penalties[-1], loop_penalties[0])


if __name__ == "__main__":
    unittest.main()
