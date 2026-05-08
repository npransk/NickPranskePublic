"""Replay buffers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Experience:
    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._items: deque[Experience] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, experience: Experience) -> None:
        self._items.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        return self._rng.sample(list(self._items), batch_size)
