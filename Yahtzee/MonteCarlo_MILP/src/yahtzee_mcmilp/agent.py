"""Agents that play the scorecard by combining the exact turn solver with a future-value model.

At the start of each turn the agent builds the terminal value of every hand:

    terminal[h] = max over open c of  points(c, h) + discount * V(next_state(c, h))

and solves the turn exactly against it (turn.py). Keep choices and the final
category choice are then just table lookups. Only the future value V
differs between agents:

- GreedyAgent: V = 0, so it maximises this turn's points and ignores the future.
- MilpAgent:   V = Monte Carlo + MILP estimate (milp.py).
"""

from __future__ import annotations

import numpy as np

from yahtzee_mcmilp.milp import MilpValueEstimator
from yahtzee_mcmilp.rules import N_HANDS, State
from yahtzee_mcmilp.turn import TurnPlan


class Agent:
    name = "agent"

    def __init__(self) -> None:
        self._plan_key: tuple | None = None
        self._plan: TurnPlan | None = None
        self._choice: np.ndarray | None = None

    def future_value(self, state: State) -> float:
        raise NotImplementedError

    def plan_turn(self, state: State) -> tuple[TurnPlan, np.ndarray]:
        """Solve the turn for this scorecard; cached until the scorecard changes."""
        if self._plan_key == state.key():
            return self._plan, self._choice
        open_cats = state.open_categories()
        values = np.full((N_HANDS, len(open_cats)), -np.inf)
        future: dict[tuple, float] = {}
        for j, c in enumerate(open_cats):
            for h in range(N_HANDS):
                points, nxt = state.apply(c, h)
                key = nxt.key()
                if key not in future:
                    future[key] = self.future_value(nxt)
                values[h, j] = points + future[key]
        best = values.argmax(axis=1)
        self._choice = np.array(open_cats)[best]
        self._plan = TurnPlan(values[np.arange(N_HANDS), best])
        self._plan_key = state.key()
        return self._plan, self._choice

    def choose_keep(self, state: State, hand: int, rolls_left: int) -> int:
        """Keep index to hold before rerolling (holding all five means stop rolling)."""
        plan, _ = self.plan_turn(state)
        return int(plan.keep[rolls_left][hand])

    def choose_category(self, state: State, hand: int) -> int:
        _, choice = self.plan_turn(state)
        return int(choice[hand])

    def expected_turn_value(self, state: State) -> float:
        return self.plan_turn(state)[0].expected


class GreedyAgent(Agent):
    """Best expected points this turn, no look-ahead."""

    name = "greedy"

    def future_value(self, state: State) -> float:
        return 0.0


class MilpAgent(Agent):
    """Exact turn play + Monte Carlo/MILP estimate of the rest of the game.

    `discount` shrinks the (optimistic) hindsight estimate toward something a
    non-clairvoyant player can actually achieve. With discount=1 the agent
    trusts the raw MILP values.
    """

    name = "mc-milp"

    def __init__(self, estimator: MilpValueEstimator | None = None, discount: float = 1.0):
        super().__init__()
        self.estimator = estimator or MilpValueEstimator()
        self.discount = discount

    def future_value(self, state: State) -> float:
        return self.discount * self.estimator.value(state)
