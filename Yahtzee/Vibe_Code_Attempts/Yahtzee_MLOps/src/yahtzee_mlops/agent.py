"""DQN agent with masked Double DQN targets."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
import random

import numpy as np
import torch
from torch import nn

from yahtzee_mlops.actions import Action, action_index, action_map
from yahtzee_mlops.game import STATE_SIZE
from yahtzee_mlops.model import DuelingDQN
from yahtzee_mlops.replay import Experience, ReplayBuffer


@dataclass
class AgentConfig:
    hidden_size: int = 256
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 256
    replay_capacity: int = 200_000
    min_replay_size: int = 5_000
    target_update_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.99995
    seed: int = 42


class DQNAgent:
    def __init__(self, config: AgentConfig, device: str | None = None) -> None:
        self.config = config
        self.actions = action_map()
        self.action_size = len(self.actions)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = random.Random(config.seed)

        self.q_network = DuelingDQN(STATE_SIZE, self.action_size, config.hidden_size).to(self.device)
        self.target_network = DuelingDQN(STATE_SIZE, self.action_size, config.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(config.replay_capacity, seed=config.seed)
        self.epsilon = config.epsilon_start
        self.steps = 0

    def select_action(self, state: np.ndarray, valid_actions: list[Action], training: bool = True) -> Action:
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()

        valid_indices = [action_index(action) for action in valid_actions]
        masked = np.full(self.action_size, -np.inf, dtype=np.float32)
        masked[valid_indices] = q_values[valid_indices]
        return self.actions[int(masked.argmax())]

    def remember(self, experience: Experience) -> None:
        self.memory.add(experience)

    def train_step(self) -> float | None:
        if len(self.memory) < max(self.config.batch_size, self.config.min_replay_size):
            return None

        batch = self.memory.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action_index for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([item.next_valid_mask for item in batch]), dtype=torch.bool, device=self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            online_next_q = self.q_network(next_states).masked_fill(~next_masks, -1e9)
            next_actions = online_next_q.argmax(dim=1)
            target_next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.gamma * target_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.config.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        if self.epsilon > self.config.epsilon_min:
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return float(loss.item())

    def checkpoint(self) -> dict[str, object]:
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "state_size": STATE_SIZE,
            "action_size": self.action_size,
        }

    def load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        self.q_network.load_state_dict(checkpoint["q_network"])  # type: ignore[arg-type]
        self.target_network.load_state_dict(checkpoint["target_network"])  # type: ignore[arg-type]
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore[arg-type]
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.steps = int(checkpoint.get("steps", self.steps))


def config_from_checkpoint(checkpoint: dict[str, object], fallback_seed: int = 42) -> AgentConfig:
    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, dict):
        return AgentConfig(seed=fallback_seed)
    allowed = {field.name for field in fields(AgentConfig)}
    values = {key: value for key, value in raw_config.items() if key in allowed}
    return AgentConfig(**values)
