from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
AGENT_IDS = {"farmer", "miner", "builder", "warrior"}


class TestOpenEnvSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest_path = ROOT / "openenv.yaml"
        self.assertTrue(self.manifest_path.exists())
        self.manifest = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))

    def test_entrypoint_uses_multi_agent_environment(self) -> None:
        self.assertEqual(self.manifest["entrypoint"], "env.multi_agent_env:MultiAgentValleyEnv")
        self.assertEqual(self.manifest["interface"]["entry_class"], "env.multi_agent_env.MultiAgentValleyEnv")

    def test_action_schema_is_nested_multi_agent_map(self) -> None:
        action_schema = self.manifest["action_schema"]
        self.assertEqual(action_schema["type"], "object")
        self.assertEqual(set(action_schema["required"]), AGENT_IDS)
        self.assertFalse(action_schema["additionalProperties"])
        self.assertEqual(set(action_schema["properties"]), AGENT_IDS)
        for agent_id in AGENT_IDS:
            self.assertEqual(action_schema["properties"][agent_id]["$ref"], "#/definitions/agent_action")

    def test_agent_action_definition_matches_action_schema(self) -> None:
        agent_action = self.manifest["definitions"]["agent_action"]
        self.assertEqual(agent_action["type"], "object")
        self.assertFalse(agent_action["additionalProperties"])
        self.assertEqual(
            set(agent_action["required"]),
            {"primary_action", "focus_resource", "cooperation_mode", "risk_posture", "rationale"},
        )
        properties = agent_action["properties"]
        self.assertEqual(
            set(properties["primary_action"]["enum"]),
            {"gather", "trade", "build", "explore", "rest", "cooperate", "compete", "defend"},
        )
        self.assertEqual(set(properties["focus_resource"]["enum"]), {"none", "food", "wood", "stone", "gold", "ore"})
        self.assertEqual(set(properties["cooperation_mode"]["enum"]), {"solo", "share", "coordinate", "protect"})
        self.assertEqual(set(properties["risk_posture"]["enum"]), {"cautious", "balanced", "aggressive"})
        self.assertEqual(properties["rationale"]["minLength"], 1)
        self.assertEqual(properties["rationale"]["maxLength"], 512)

    def test_sample_action_contains_all_agents(self) -> None:
        sample = self.manifest["examples"]["sample_action"]["value"]
        self.assertEqual(set(sample), AGENT_IDS)
        for agent_action in sample.values():
            self.assertEqual(
                set(agent_action),
                {"primary_action", "focus_resource", "cooperation_mode", "risk_posture", "rationale"},
            )


if __name__ == "__main__":
    unittest.main()
