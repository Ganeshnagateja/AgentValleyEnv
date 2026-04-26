from __future__ import annotations

import unittest

from env.action_space import action_to_index, index_to_action, list_all_actions, random_action
from env.schemas import Action


class TestActionSpace(unittest.TestCase):
    def test_action_index_roundtrip(self) -> None:
        for raw in list_all_actions():
            action = Action(**raw)
            roundtrip = index_to_action(action_to_index(action))
            self.assertEqual(action.primary_action, roundtrip.primary_action)
            self.assertEqual(action.focus_resource, roundtrip.focus_resource)
            self.assertEqual(action.cooperation_mode, roundtrip.cooperation_mode)
            self.assertEqual(action.risk_posture, roundtrip.risk_posture)

    def test_random_action_is_schema_valid(self) -> None:
        action = random_action()
        self.assertIsInstance(action, Action)
        self.assertGreaterEqual(action_to_index(action), 0)
        self.assertTrue(action.rationale)


if __name__ == "__main__":
    unittest.main()
