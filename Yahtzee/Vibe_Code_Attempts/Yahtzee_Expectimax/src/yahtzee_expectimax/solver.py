from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from yahtzee_expectimax.constants import ALL_USED_MASK, CATEGORIES, UPPER_BONUS, UPPER_BONUS_TARGET, YAHTZEE_BONUS
from yahtzee_expectimax.dice import Counts, add_counts, legal_keeps, roll_distribution
from yahtzee_expectimax.scoring import is_yahtzee, score_counts


@dataclass(frozen=True)
class Decision:
    kind: str
    payload: Counts | str
    expected_value: float


class ExpectimaxSolver:
    """Exact expectimax policy over scorecard state and dice-count states.

    State tracks the used score categories, upper subtotal capped at 63, and
    whether Yahtzee bonuses are enabled. It models the common bonus rule and a
    simplified joker rule by allowing any open category after bonus Yahtzees.
    """

    def __init__(self) -> None:
        self.categories = CATEGORIES

    def best_action(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts, rolls_left: int) -> Decision:
        upper = min(upper_total, UPPER_BONUS_TARGET)
        if rolls_left <= 0:
            category, value = self._best_score_action(used_mask, upper, yahtzee_bonus_enabled, dice)
            return Decision("score", category, value)

        best_keep: Counts | None = None
        best_value = float("-inf")
        for keep in legal_keeps(dice):
            value = self._expected_after_keep(used_mask, upper, yahtzee_bonus_enabled, keep, rolls_left)
            if value > best_value:
                best_keep = keep
                best_value = value

        score_category, score_value = self._best_score_action(used_mask, upper, yahtzee_bonus_enabled, dice)
        if score_value >= best_value:
            return Decision("score", score_category, score_value)
        return Decision("keep", best_keep or (0, 0, 0, 0, 0, 0), best_value)

    def cache_info(self) -> dict[str, object]:
        return {
            "game_value": self._game_value.cache_info(),
            "turn_value": self._turn_value.cache_info(),
            "expected_after_keep": self._expected_after_keep.cache_info(),
        }

    @lru_cache(maxsize=None)
    def _game_value(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool) -> float:
        if used_mask == ALL_USED_MASK:
            return 0.0
        expected = 0.0
        for dice, probability in roll_distribution(5):
            expected += probability * self._turn_value(used_mask, upper_total, yahtzee_bonus_enabled, dice, 2)
        return expected

    @lru_cache(maxsize=None)
    def _turn_value(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts, rolls_left: int) -> float:
        if rolls_left <= 0:
            return self._best_score_action(used_mask, upper_total, yahtzee_bonus_enabled, dice)[1]

        best_roll = max(
            self._expected_after_keep(used_mask, upper_total, yahtzee_bonus_enabled, keep, rolls_left)
            for keep in legal_keeps(dice)
        )
        best_score = self._best_score_action(used_mask, upper_total, yahtzee_bonus_enabled, dice)[1]
        return max(best_roll, best_score)

    @lru_cache(maxsize=None)
    def _expected_after_keep(
        self,
        used_mask: int,
        upper_total: int,
        yahtzee_bonus_enabled: bool,
        keep: Counts,
        rolls_left: int,
    ) -> float:
        reroll_count = 5 - sum(keep)
        expected = 0.0
        for rolled, probability in roll_distribution(reroll_count):
            next_dice = add_counts(keep, rolled)
            expected += probability * self._turn_value(used_mask, upper_total, yahtzee_bonus_enabled, next_dice, rolls_left - 1)
        return expected

    @lru_cache(maxsize=None)
    def _best_score_action(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts) -> tuple[str, float]:
        best_category = ""
        best_value = float("-inf")
        for index, category in enumerate(CATEGORIES):
            if used_mask & (1 << index):
                continue

            immediate = score_counts(category, dice)
            new_upper = upper_total
            if index < 6:
                new_upper = min(UPPER_BONUS_TARGET, upper_total + immediate)
                if upper_total < UPPER_BONUS_TARGET <= new_upper:
                    immediate += UPPER_BONUS

            new_yahtzee_bonus_enabled = yahtzee_bonus_enabled or (category == "yahtzee" and immediate >= 50)
            if category != "yahtzee" and yahtzee_bonus_enabled and is_yahtzee(dice):
                immediate += YAHTZEE_BONUS

            value = immediate + self._game_value(used_mask | (1 << index), new_upper, new_yahtzee_bonus_enabled)
            if value > best_value:
                best_category = category
                best_value = value
        return best_category, best_value
