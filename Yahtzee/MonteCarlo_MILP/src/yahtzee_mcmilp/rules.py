"""Yahtzee rules and precomputed dice tables.

Dice are represented as face counts (a hand of [2, 2, 5, 5, 6] is the count
vector (0, 2, 0, 0, 2, 1)), which shrinks 7,776 ordered rolls down to 252
distinct hands. Every table the solver needs is built once at import time.

Rules match the rest of this repo so scores are comparable:
- 13 categories, +35 upper bonus at 63, +100 per extra Yahtzee once the
  Yahtzee box holds 50.
- No joker rules (same simplification as the DQN and expectimax attempts).
"""

from __future__ import annotations

from itertools import product
from math import factorial

import numpy as np

CATEGORIES = (
    "ones", "twos", "threes", "fours", "fives", "sixes",
    "three_of_kind", "four_of_kind", "full_house",
    "small_straight", "large_straight", "yahtzee", "chance",
)
N_CATEGORIES = 13
UPPER = tuple(range(6))
YAHTZEE = 11
UPPER_TARGET = 63
UPPER_BONUS = 35
YAHTZEE_BONUS = 100
ALL_USED = (1 << N_CATEGORIES) - 1


def _count_vectors(total: int) -> list[tuple[int, ...]]:
    return [c for c in product(range(total + 1), repeat=6) if sum(c) == total]


# All 252 five-dice hands and all 462 keeps (0-5 dice held), as count vectors.
HANDS = np.array(_count_vectors(5), dtype=np.int8)
KEEPS = np.array([c for k in range(6) for c in _count_vectors(k)], dtype=np.int8)
N_HANDS = len(HANDS)
N_KEEPS = len(KEEPS)
HAND_INDEX = {tuple(h): i for i, h in enumerate(HANDS.tolist())}
KEEP_INDEX = {tuple(k): i for i, k in enumerate(KEEPS.tolist())}


def score_counts(counts, category: int) -> int:
    """Base category score for a hand (no bonuses)."""
    counts = [int(c) for c in counts]
    total = sum((face + 1) * c for face, c in enumerate(counts))
    faces = {face + 1 for face, c in enumerate(counts) if c}
    if category < 6:
        return counts[category] * (category + 1)
    if category == 6:
        return total if max(counts) >= 3 else 0
    if category == 7:
        return total if max(counts) >= 4 else 0
    if category == 8:
        return 25 if sorted(c for c in counts if c) == [2, 3] else 0
    if category == 9:
        return 30 if any(run <= faces for run in ({1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6})) else 0
    if category == 10:
        return 40 if faces in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0
    if category == 11:
        return 50 if max(counts) == 5 else 0
    if category == 12:
        return total
    raise ValueError(f"unknown category {category}")


def _multinomial_probs(n_dice: int) -> list[tuple[np.ndarray, float]]:
    out = []
    for counts in _count_vectors(n_dice):
        ways = factorial(n_dice)
        for c in counts:
            ways //= factorial(c)
        out.append((np.array(counts, dtype=np.int8), ways / 6**n_dice))
    return out


SCORE_TABLE = np.array(
    [[score_counts(h, c) for c in range(N_CATEGORIES)] for h in HANDS], dtype=np.int16
)
IS_YAHTZEE = HANDS.max(axis=1) == 5

# TRANSITION[k, h] = P(hand h | keep k and reroll the other dice).
TRANSITION = np.zeros((N_KEEPS, N_HANDS))
for _k, _keep in enumerate(KEEPS):
    for _rolled, _p in _multinomial_probs(5 - int(_keep.sum())):
        TRANSITION[_k, HAND_INDEX[tuple((_keep + _rolled).tolist())]] += _p

# Distribution of the opening roll of a turn (keep nothing, roll five).
FIRST_ROLL = TRANSITION[KEEP_INDEX[(0,) * 6]].copy()

# LEGAL_KEEP[h, k] is True when keep k is a sub-multiset of hand h.
LEGAL_KEEP = (KEEPS[None, :, :] <= HANDS[:, None, :]).all(axis=2)


def hand_index(dice) -> int:
    """Index of a hand given as five die faces (1-6)."""
    counts = [0] * 6
    for d in dice:
        counts[int(d) - 1] += 1
    return HAND_INDEX[tuple(counts)]


def keep_from_faces(faces) -> int:
    counts = [0] * 6
    for d in faces:
        counts[int(d) - 1] += 1
    return KEEP_INDEX[tuple(counts)]


def faces_from_counts(counts) -> list[int]:
    return [face + 1 for face, c in enumerate(counts) for _ in range(int(c))]


class State:
    """Scorecard state between turns: everything that matters for the future.

    `upper` is capped at 63 because nothing above the bonus line changes
    future value. `ybonus` is True once the Yahtzee box holds 50.
    """

    __slots__ = ("used", "upper", "ybonus")

    def __init__(self, used: int = 0, upper: int = 0, ybonus: bool = False):
        self.used = used
        self.upper = upper
        self.ybonus = ybonus

    def key(self) -> tuple[int, int, bool]:
        return (self.used, self.upper, self.ybonus)

    def open_categories(self) -> list[int]:
        return [c for c in range(N_CATEGORIES) if not self.used >> c & 1]

    def is_over(self) -> bool:
        return self.used == ALL_USED

    def apply(self, category: int, hand: int) -> tuple[int, "State"]:
        """Score `hand` in `category`; return (points gained incl. bonuses, next state)."""
        if self.used >> category & 1:
            raise ValueError(f"{CATEGORIES[category]} already used")
        base = int(SCORE_TABLE[hand, category])
        points = base
        if self.ybonus and IS_YAHTZEE[hand]:
            points += YAHTZEE_BONUS
        upper = self.upper
        if category < 6:
            # Only the box score counts toward 63, never the Yahtzee bonus.
            new_upper = min(UPPER_TARGET, upper + base)
            if upper < UPPER_TARGET <= new_upper:
                points += UPPER_BONUS
            upper = new_upper
        ybonus = self.ybonus or (category == YAHTZEE and IS_YAHTZEE[hand])
        return points, State(self.used | 1 << category, upper, bool(ybonus))

    def __repr__(self) -> str:
        open_names = [CATEGORIES[c] for c in self.open_categories()]
        return f"State(upper={self.upper}, ybonus={self.ybonus}, open={open_names})"
