"""Neural network models."""

from __future__ import annotations

import torch
from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.features(states)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
