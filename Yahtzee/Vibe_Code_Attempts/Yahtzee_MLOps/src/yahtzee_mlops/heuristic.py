"""Fast baseline policies for play, benchmarking, and warm starts."""

from __future__ import annotations

import random

from yahtzee_mlops.actions import Action
from yahtzee_mlops.constants import CATEGORIES, UPPER_CATEGORIES
from yahtzee_mlops.game import YahtzeeGame
from yahtzee_mlops.scoring import face_counts, score_category


def _available_categories(game: YahtzeeGame) -> list[str]:
    return [category for category in CATEGORIES if game.scorecard[category] is None]


def _upper_need(game: YahtzeeGame) -> int:
    return max(0, 63 - sum(game.scorecard[category] or 0 for category in UPPER_CATEGORIES))


def choose_score_category(game: YahtzeeGame) -> str:
    available = _available_categories(game)
    scored = [(category, score_category(category, game.dice)) for category in available]
    upper_need = _upper_need(game)

    def value(item: tuple[str, int]) -> float:
        category, score = item
        if category in UPPER_CATEGORIES:
            face = UPPER_CATEGORIES.index(category) + 1
            target = face * 3
            bonus_pressure = 8.0 if upper_need <= 18 else 4.0
            return score + (bonus_pressure if score >= target else -2.0)
        if category == "yahtzee":
            return score + (12.0 if score else -5.0)
        if category == "large_straight":
            return score + (8.0 if score else -3.0)
        if category == "small_straight":
            return score + (4.0 if score else -2.0)
        if category == "chance":
            return score - (5.0 if len(available) > 4 else 0.0)
        return float(score)

    nonzero = [item for item in scored if item[1] > 0]
    if nonzero:
        return max(nonzero, key=value)[0]

    sacrifice_order = [
        "ones",
        "twos",
        "three_of_kind",
        "four_of_kind",
        "chance",
        "full_house",
        "small_straight",
        "large_straight",
        "yahtzee",
        "threes",
        "fours",
        "fives",
        "sixes",
    ]
    return next(category for category in sacrifice_order if category in available)


def choose_keep_counts(game: YahtzeeGame) -> tuple[int, ...]:
    counts = face_counts(game.dice)
    available = set(_available_categories(game))

    if "yahtzee" in available or "four_of_kind" in available or "three_of_kind" in available:
        best_face = max(range(1, 7), key=lambda face: (counts[face - 1], face))
        if counts[best_face - 1] >= 2:
            return tuple(count if face == best_face else 0 for face, count in enumerate(counts, start=1))

    faces = {face for face, count in enumerate(counts, start=1) if count}
    straight_targets = []
    if "large_straight" in available:
        straight_targets.extend([{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}])
    if "small_straight" in available:
        straight_targets.extend([{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}])
    if straight_targets:
        target = max(straight_targets, key=lambda run: len(run & faces))
        if len(target & faces) >= 3:
            return tuple(1 if face in target and counts[face - 1] else 0 for face in range(1, 7))

    upper_faces = [UPPER_CATEGORIES.index(category) + 1 for category in available if category in UPPER_CATEGORIES]
    if upper_faces:
        best_face = max(upper_faces, key=lambda face: (counts[face - 1] * face, face))
        if counts[best_face - 1] > 0:
            return tuple(count if face == best_face else 0 for face, count in enumerate(counts, start=1))

    best_face = max(range(1, 7), key=lambda face: (counts[face - 1], face))
    return tuple(count if face == best_face else 0 for face, count in enumerate(counts, start=1))


class HeuristicPlayer:
    def select_action(self, game: YahtzeeGame) -> Action:
        if game.first_roll_of_turn:
            return ("roll", (0, 0, 0, 0, 0, 0))
        if game.roll_count < 3:
            keep = choose_keep_counts(game)
            if sum(keep) < 5:
                return ("roll", keep)
        return ("score", choose_score_category(game))


class RandomPlayer:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def select_action(self, game: YahtzeeGame) -> Action:
        return self.rng.choice(game.legal_actions())


def describe_action(action: Action, dice: tuple[int, ...] | None = None, score_delta: int | None = None) -> str:
    action_type, payload = action
    if action_type == "roll":
        keep = payload
        labels = [f"{count} x {face}" for face, count in enumerate(keep, start=1) if count]  # type: ignore[arg-type]
        kept = ", ".join(labels) if labels else "nothing"
        suffix = f" -> rolled {dice}" if dice else ""
        return f"Kept {kept}{suffix}"
    category = str(payload).replace("_", " ").title()
    suffix = f" for {score_delta} points" if score_delta is not None else ""
    return f"Scored {category}{suffix}"
