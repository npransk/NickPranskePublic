"""Yahtzee scoring rules."""

from __future__ import annotations

from collections import Counter


def face_counts(dice: tuple[int, ...]) -> tuple[int, ...]:
    counts = [0] * 6
    for die in dice:
        if die < 1 or die > 6:
            raise ValueError(f"Invalid die value: {die}")
        counts[die - 1] += 1
    return tuple(counts)


def score_category(category: str, dice: tuple[int, ...]) -> int:
    counts = face_counts(dice)
    dice_sum = sum(dice)

    if category in {"ones", "twos", "threes", "fours", "fives", "sixes"}:
        face = ("ones", "twos", "threes", "fours", "fives", "sixes").index(category) + 1
        return counts[face - 1] * face
    if category == "three_of_kind":
        return dice_sum if max(counts) >= 3 else 0
    if category == "four_of_kind":
        return dice_sum if max(counts) >= 4 else 0
    if category == "full_house":
        nonzero = sorted(count for count in counts if count)
        return 25 if nonzero == [2, 3] else 0
    if category == "small_straight":
        faces = {face for face, count in enumerate(counts, start=1) if count}
        return 30 if any(run <= faces for run in ({1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6})) else 0
    if category == "large_straight":
        return 40 if set(dice) in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0
    if category == "yahtzee":
        return 50 if max(counts) == 5 else 0
    if category == "chance":
        return dice_sum
    raise ValueError(f"Unknown category: {category}")


def is_yahtzee(dice: tuple[int, ...]) -> bool:
    return len(Counter(dice)) == 1
