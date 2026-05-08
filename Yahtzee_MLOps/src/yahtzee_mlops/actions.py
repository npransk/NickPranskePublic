"""Action encoding helpers.

Roll actions keep dice by face count instead of physical die position. This
removes duplicate actions caused by permuting identical dice.
"""

from __future__ import annotations

from functools import lru_cache

from yahtzee_mlops.constants import CATEGORIES, NUM_DICE, NUM_FACES

Action = tuple[str, tuple[int, ...] | str]


def _count_patterns(total: int, bins: int) -> list[tuple[int, ...]]:
    if bins == 1:
        return [(total,)]
    patterns: list[tuple[int, ...]] = []
    for value in range(total + 1):
        for suffix in _count_patterns(total - value, bins - 1):
            patterns.append((value, *suffix))
    return patterns


@lru_cache(maxsize=1)
def keep_count_actions() -> tuple[tuple[int, ...], ...]:
    patterns: list[tuple[int, ...]] = []
    for kept in range(NUM_DICE + 1):
        patterns.extend(_count_patterns(kept, NUM_FACES))
    return tuple(patterns)


@lru_cache(maxsize=1)
def action_map() -> tuple[Action, ...]:
    roll_actions: list[Action] = [("roll", counts) for counts in keep_count_actions()]
    score_actions: list[Action] = [("score", category) for category in CATEGORIES]
    return tuple(roll_actions + score_actions)


ACTION_TO_INDEX = {action: index for index, action in enumerate(action_map())}


def action_index(action: Action) -> int:
    return ACTION_TO_INDEX[action]


def valid_action_mask(valid_actions: list[Action]) -> list[bool]:
    valid = set(valid_actions)
    return [action in valid for action in action_map()]
