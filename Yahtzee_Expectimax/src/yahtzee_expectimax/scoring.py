from __future__ import annotations

from yahtzee_expectimax.constants import CATEGORIES
from yahtzee_expectimax.dice import Counts


def score_counts(category: str, counts: Counts) -> int:
    dice_sum = sum((face + 1) * count for face, count in enumerate(counts))

    if category in CATEGORIES[:6]:
        face = CATEGORIES.index(category) + 1
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
        faces = {face for face, count in enumerate(counts, start=1) if count}
        return 40 if faces in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0
    if category == "yahtzee":
        return 50 if max(counts) == 5 else 0
    if category == "chance":
        return dice_sum
    raise ValueError(f"Unknown category: {category}")


def is_yahtzee(counts: Counts) -> bool:
    return max(counts) == 5
