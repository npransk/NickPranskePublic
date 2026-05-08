from __future__ import annotations

from functools import lru_cache
from math import factorial
from random import Random

from yahtzee_expectimax.constants import N_DICE, N_FACES

Counts = tuple[int, int, int, int, int, int]


def count_patterns(total: int, bins: int = N_FACES) -> tuple[Counts, ...]:
    def rec(remaining: int, slots: int) -> list[tuple[int, ...]]:
        if slots == 1:
            return [(remaining,)]
        out: list[tuple[int, ...]] = []
        for value in range(remaining + 1):
            for suffix in rec(remaining - value, slots - 1):
                out.append((value, *suffix))
        return out

    return tuple(rec(total, bins))  # type: ignore[return-value]


ALL_DICE_COUNTS = count_patterns(N_DICE)
ALL_KEEP_COUNTS = tuple(pattern for kept in range(N_DICE + 1) for pattern in count_patterns(kept))


@lru_cache(maxsize=None)
def roll_distribution(n_dice: int) -> tuple[tuple[Counts, float], ...]:
    outcomes: list[tuple[Counts, float]] = []
    total = N_FACES**n_dice
    for counts in count_patterns(n_dice):
        ways = factorial(n_dice)
        for count in counts:
            ways //= factorial(count)
        outcomes.append((counts, ways / total))
    return tuple(outcomes)


def add_counts(left: Counts, right: Counts) -> Counts:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def can_keep(keep: Counts, dice: Counts) -> bool:
    return all(k <= d for k, d in zip(keep, dice))


def legal_keeps(dice: Counts) -> tuple[Counts, ...]:
    return tuple(keep for keep in ALL_KEEP_COUNTS if can_keep(keep, dice))


def roll_all(rng: Random) -> Counts:
    counts = [0] * N_FACES
    for _ in range(N_DICE):
        counts[rng.randint(0, N_FACES - 1)] += 1
    return tuple(counts)  # type: ignore[return-value]


def reroll_from_keep(keep: Counts, rng: Random) -> Counts:
    counts = list(keep)
    for _ in range(N_DICE - sum(keep)):
        counts[rng.randint(0, N_FACES - 1)] += 1
    return tuple(counts)  # type: ignore[return-value]


def counts_to_dice(counts: Counts) -> tuple[int, ...]:
    dice: list[int] = []
    for face, count in enumerate(counts, start=1):
        dice.extend([face] * count)
    return tuple(dice)
