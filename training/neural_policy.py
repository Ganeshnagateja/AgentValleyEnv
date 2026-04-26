"""Small CPU-friendly PyTorch policy for AgentValleyEnv."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from env.action_space import action_count, index_to_action
from training.feature_encoder import FEATURE_DIM, encode_observation


class NeuralPolicy(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, action_dim: int | None = None, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim or action_count()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def observation_tensor(observation: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(encode_observation(observation), dtype=torch.float32).unsqueeze(0)


def distribution_for_observation(policy: NeuralPolicy, observation: Mapping[str, Any]) -> Categorical:
    logits = policy(observation_tensor(observation))
    return Categorical(logits=logits)


def select_action(policy: NeuralPolicy, observation: Mapping[str, Any], deterministic: bool = True) -> tuple[int, dict[str, Any]]:
    with torch.no_grad():
        dist = distribution_for_observation(policy, observation)
        if deterministic:
            action_index = int(torch.argmax(dist.logits, dim=-1).item())
        else:
            action_index = int(dist.sample().item())
    return action_index, index_to_action(action_index).model_dump(mode="json")


def save_checkpoint(
    path: Path,
    policy: NeuralPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": policy.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, hidden_dim: int = 64) -> tuple[NeuralPolicy, dict[str, Any]]:
    policy = NeuralPolicy(hidden_dim=hidden_dim)
    if not path.exists():
        return policy, {"checkpoint_found": False}
    payload = torch.load(path, map_location="cpu")
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()
    metadata = dict(payload.get("metadata") or {})
    metadata["checkpoint_found"] = True
    return policy, metadata
