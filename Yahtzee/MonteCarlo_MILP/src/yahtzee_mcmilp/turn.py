"""Exact within-turn expectimax.

Given the value of ending the turn with each hand (`terminal[h]`), this solves
the three-roll turn exactly by backward induction over the 252 hands and 462
keeps. Everything is vectorised with numpy, so a full turn solve takes well
under a millisecond.

`terminal` is where the long-horizon model plugs in: immediate points plus
the estimated future value of the scorecard you would be left with. With a
zero future value this is a greedy "best score this turn" player; with the
Monte Carlo + MILP estimate it becomes the full agent.
"""

from __future__ import annotations

import numpy as np

from yahtzee_mcmilp.rules import FIRST_ROLL, LEGAL_KEEP, TRANSITION

_NEG = -1e18


def _best_keep(values_after_roll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each hand, the value and index of the best keep given next-roll values."""
    keep_values = TRANSITION @ values_after_roll  # (462,)
    masked = np.where(LEGAL_KEEP, keep_values[None, :], _NEG)  # (252, 462)
    best = masked.argmax(axis=1)
    return masked[np.arange(len(best)), best], best


class TurnPlan:
    """Solved turn: best keep for every (hand, rolls left) and the turn's expected value.

    value[r][h] is the expected value holding hand h with r rerolls left.
    keep[r][h] is the keep index to use in that situation (r = 1 or 2).
    Keeping all five dice is one of the keeps, so "stop rolling" is covered.
    """

    def __init__(self, terminal: np.ndarray):
        self.terminal = terminal
        v1, k1 = _best_keep(terminal)
        v2, k2 = _best_keep(v1)
        self.value = {0: terminal, 1: v1, 2: v2}
        self.keep = {1: k1, 2: k2}
        self.expected = float(FIRST_ROLL @ v2)

