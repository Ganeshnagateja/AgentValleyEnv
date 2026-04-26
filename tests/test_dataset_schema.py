from __future__ import annotations

import json
import unittest
from pathlib import Path

from env.schemas import Observation


ROOT = Path(__file__).resolve().parent.parent


class TestDatasetSchema(unittest.TestCase):
    def test_all_dataset_observations_validate(self) -> None:
        for path in sorted((ROOT / "datasets").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for episode in payload["episodes"]:
                for step in episode["steps"]:
                    with self.subTest(dataset=path.name, episode=episode.get("episode_id"), tick=step.get("tick")):
                        obs = Observation(**step)
                        self.assertEqual(obs.region, step["region"])


if __name__ == "__main__":
    unittest.main()
