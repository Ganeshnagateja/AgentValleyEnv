from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch

from env.action_space import random_action
from env.agents import AGENT_IDS, get_partial_obs
from env.multi_agent_env import MultiAgentValleyEnv
from training.ma_grpo_train import MAGRPOConfig, MAGRPOTrainer


def coordinated_actions() -> dict[str, dict]:
    return {
        "farmer": {
            "primary_action": "gather",
            "focus_resource": "food",
            "cooperation_mode": "share",
            "risk_posture": "balanced",
            "rationale": "Food is the farmer contribution to the team plan.",
        },
        "miner": {
            "primary_action": "gather",
            "focus_resource": "ore",
            "cooperation_mode": "coordinate",
            "risk_posture": "balanced",
            "rationale": "Ore supports the shared building plan.",
        },
        "builder": {
            "primary_action": "build",
            "focus_resource": "stone",
            "cooperation_mode": "coordinate",
            "risk_posture": "cautious",
            "rationale": "Stone building improves team resilience.",
        },
        "warrior": {
            "primary_action": "defend",
            "focus_resource": "none",
            "cooperation_mode": "protect",
            "risk_posture": "cautious",
            "rationale": "Defense protects the rest of the team.",
        },
    }


class TestMultiAgentValleyEnv(unittest.TestCase):
    def test_reset_returns_all_agents(self) -> None:
        env = MultiAgentValleyEnv(difficulty="hard", seed=42)
        obs = env.reset()
        self.assertEqual(set(obs), set(AGENT_IDS))
        for agent_id, agent_obs in obs.items():
            self.assertEqual(agent_obs["agent_id"], agent_id)
            self.assertIn("agent_role", agent_obs)
            self.assertIn("agent_home_region", agent_obs)

    def test_partial_obs_noise(self) -> None:
        full_obs = {
            "region": "farmland",
            "food_supply": 0.5,
            "wood_supply": 0.5,
            "stone_supply": 0.5,
            "gold_supply": 0.5,
            "ore_supply": 0.5,
        }
        farmer_obs = get_partial_obs(full_obs, "farmer", seed=42)
        miner_obs = get_partial_obs(full_obs, "miner", seed=42)
        self.assertEqual(farmer_obs["ore_supply"], 0.5)
        self.assertNotEqual(miner_obs["ore_supply"], 0.5)

    def test_step_requires_all_agents(self) -> None:
        env = MultiAgentValleyEnv(seed=42)
        env.reset()
        actions = coordinated_actions()
        actions.pop("warrior")
        with self.assertRaises(ValueError):
            env.step(actions)

    def test_step_rejects_unknown_agent_ids(self) -> None:
        env = MultiAgentValleyEnv(seed=42)
        env.reset()
        actions = coordinated_actions()
        actions["extra_agent"] = coordinated_actions()["farmer"]
        with self.assertRaises(ValueError):
            env.step(actions)

    def test_step_returns_all_agent_rewards(self) -> None:
        env = MultiAgentValleyEnv(seed=42)
        env.reset()
        _obs, rewards, dones, info = env.step(coordinated_actions())
        self.assertEqual(set(rewards), set(AGENT_IDS))
        self.assertTrue(all(isinstance(value, float) for value in rewards.values()))
        self.assertTrue(set(AGENT_IDS).issubset(dones))
        self.assertIn("reward_components_by_agent", info)

    def test_cooperation_bonus_warrior_plus_farmer(self) -> None:
        env = MultiAgentValleyEnv(seed=42)
        env.reset()
        _obs, _rewards, _dones, info = env.step(coordinated_actions())
        self.assertGreater(info["cooperation_bonus"], 0.0)

    def test_conflict_penalty_pile_on(self) -> None:
        env = MultiAgentValleyEnv(seed=42)
        env.reset()
        pile_on = {
            agent_id: {
                "primary_action": "gather",
                "focus_resource": "food",
                "cooperation_mode": "solo",
                "risk_posture": "aggressive",
                "rationale": "This uncoordinated baseline piles onto one action.",
            }
            for agent_id in AGENT_IDS
        }
        _obs, _rewards, _dones, info = env.step(pile_on)
        self.assertLess(info["conflict_penalty"], 0.0)

    def test_coordinated_actions_outscore_selfish_pile_on(self) -> None:
        coordinated_env = MultiAgentValleyEnv(difficulty="easy", seed=42)
        coordinated_env.reset()
        _obs, coordinated_rewards, _dones, coordinated_info = coordinated_env.step(coordinated_actions())

        selfish_env = MultiAgentValleyEnv(difficulty="easy", seed=42)
        selfish_env.reset()
        selfish_actions = {
            agent_id: {
                "primary_action": "gather",
                "focus_resource": "food",
                "cooperation_mode": "solo",
                "risk_posture": "aggressive",
                "rationale": "This baseline ignores complementary roles.",
            }
            for agent_id in AGENT_IDS
        }
        _obs, selfish_rewards, _dones, selfish_info = selfish_env.step(selfish_actions)

        self.assertGreater(sum(coordinated_rewards.values()), sum(selfish_rewards.values()))
        self.assertGreater(coordinated_info["cooperation_bonus"], selfish_info["cooperation_bonus"])

    def test_done_propagates_to_all_agents(self) -> None:
        env = MultiAgentValleyEnv(difficulty="easy", seed=42)
        env.reset()
        done = False
        dones = {}
        for _ in range(10):
            _obs, _rewards, dones, _info = env.step(coordinated_actions())
            done = bool(dones.get("__all__"))
            if done:
                break
        self.assertTrue(done)
        self.assertTrue(all(dones[agent_id] for agent_id in AGENT_IDS))

    def test_lead_agent_warrior_on_high_threat(self) -> None:
        env = MultiAgentValleyEnv(seed=42)
        env.reset()
        obs = {"threat_level": 0.9, "food_supply": 0.8, "defense_readiness": 0.8}
        self.assertEqual(env._pick_lead_agent(obs, coordinated_actions()), "warrior")

    def test_episode_runs_to_completion(self) -> None:
        env = MultiAgentValleyEnv(difficulty="easy", seed=42)
        env.reset()
        rng = random.Random(42)
        done = False
        for _ in range(20):
            actions = {agent_id: random_action(rng).model_dump(mode="json") for agent_id in AGENT_IDS}
            _obs, _rewards, dones, _info = env.step(actions)
            done = bool(dones.get("__all__"))
            if done:
                break
        self.assertTrue(done)

    def test_ma_grpo_trainer_updates_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MAGRPOTrainer(
                MAGRPOConfig(episodes=1, difficulty="easy", seed=42, group_size=2, artifact_dir=Path(tmp))
            )
            before = {
                agent_id: [param.detach().clone() for param in trainer.policies[agent_id].parameters()]
                for agent_id in AGENT_IDS
            }
            metrics = trainer.train()
            self.assertTrue(trainer.metrics_path.exists())
            self.assertEqual(metrics[0]["mode"], "multi_agent_grpo")
            self.assertIn("total_team_reward", metrics[0])
            for agent_id in AGENT_IDS:
                after = [param.detach().clone() for param in trainer.policies[agent_id].parameters()]
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before[agent_id], after)))
                self.assertIn(agent_id, metrics[0]["agent_metrics"])


if __name__ == "__main__":
    unittest.main()
