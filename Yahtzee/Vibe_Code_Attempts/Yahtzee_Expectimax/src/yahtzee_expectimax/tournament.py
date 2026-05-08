from __future__ import annotations

from functools import lru_cache

from yahtzee_expectimax.constants import CATEGORIES, UPPER_BONUS, UPPER_BONUS_TARGET, YAHTZEE_BONUS
from yahtzee_expectimax.dice import Counts, add_counts, can_keep, legal_keeps, roll_distribution
from yahtzee_expectimax.scoring import is_yahtzee, score_counts


UPPER_TARGETS = (3, 6, 9, 12, 15, 18)
LOWER_BASELINE = {
    "three_of_kind": 21,
    "four_of_kind": 13,
    "full_house": 22,
    "small_straight": 24,
    "large_straight": 18,
    "yahtzee": 12,
    "chance": 23,
}


def _upper_total_from_scores(scores: tuple[int, ...]) -> int:
    return sum(scores[:6])


def score_utility(category: str, raw_score: int, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts) -> float:
    index = CATEGORIES.index(category)
    utility = float(raw_score)
    open_count = 13 - used_mask.bit_count()

    if category in CATEGORIES[:6]:
        target = UPPER_TARGETS[index]
        face = index + 1
        progress_before = min(upper_total, UPPER_BONUS_TARGET)
        progress_after = min(upper_total + raw_score, UPPER_BONUS_TARGET)
        utility += (progress_after - progress_before) * 1.25
        if raw_score >= target:
            utility += 7.0 + face * 0.55
        elif upper_total < UPPER_BONUS_TARGET:
            utility -= (target - raw_score) * 1.45
        if progress_before < UPPER_BONUS_TARGET <= progress_after:
            utility += UPPER_BONUS
    else:
        utility += raw_score - LOWER_BASELINE.get(category, 0)

    if category == "chance" and open_count > 5:
        utility -= 11.0
    if category == "yahtzee":
        utility += 35.0 if raw_score == 50 else (-18.0 if open_count > 3 else -3.0)
    if category == "large_straight":
        utility += 7.0 if raw_score == 40 else (-5.0 if open_count > 4 else 0.0)
    if category == "small_straight":
        utility += 4.0 if raw_score == 30 else (-3.0 if open_count > 4 else 0.0)
    if category == "full_house":
        utility += 2.0 if raw_score == 25 else (-2.0 if open_count > 4 else 0.0)
    if category != "yahtzee" and yahtzee_bonus_enabled and is_yahtzee(dice):
        utility += YAHTZEE_BONUS
    return utility


class TournamentPolicy:
    """Fast high-score policy using exact turn search and tuned scorecard utility."""

    def best_score_category(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts) -> tuple[str, float]:
        best_category = ""
        best_value = float("-inf")
        for index, category in enumerate(CATEGORIES):
            if used_mask & (1 << index):
                continue
            raw = score_counts(category, dice)
            value = score_utility(category, raw, used_mask, upper_total, yahtzee_bonus_enabled, dice)
            if value > best_value:
                best_category = category
                best_value = value
        return best_category, best_value

    def best_keep(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts, rolls_left: int) -> tuple[Counts, float]:
        best_keep = (0, 0, 0, 0, 0, 0)
        best_value = float("-inf")
        for keep in candidate_keeps(dice, used_mask):
            value = self._expected_keep_value(used_mask, upper_total, yahtzee_bonus_enabled, keep, rolls_left)
            if value > best_value:
                best_keep = keep
                best_value = value
        return best_keep, best_value

    def choose(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts, rolls_left: int) -> tuple[str, Counts | str]:
        if rolls_left <= 0:
            return "score", self.best_score_category(used_mask, upper_total, yahtzee_bonus_enabled, dice)[0]

        category, score_value = self.best_score_category(used_mask, upper_total, yahtzee_bonus_enabled, dice)
        keep, keep_value = self.best_keep(used_mask, upper_total, yahtzee_bonus_enabled, dice, rolls_left)

        # Only stop early for genuinely strong made hands.
        if score_value >= keep_value + 2.0:
            return "score", category
        return "keep", keep

    @lru_cache(maxsize=None)
    def _turn_value(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts, rolls_left: int) -> float:
        if rolls_left <= 0:
            return self.best_score_category(used_mask, upper_total, yahtzee_bonus_enabled, dice)[1]
        return max(
            self.best_score_category(used_mask, upper_total, yahtzee_bonus_enabled, dice)[1],
            max(self._expected_keep_value(used_mask, upper_total, yahtzee_bonus_enabled, keep, rolls_left) for keep in legal_keeps(dice)),
        )

    @lru_cache(maxsize=None)
    def _expected_keep_value(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, keep: Counts, rolls_left: int) -> float:
        expected = 0.0
        for rolled, probability in roll_distribution(5 - sum(keep)):
            expected += probability * self._turn_value(
                used_mask,
                upper_total,
                yahtzee_bonus_enabled,
                add_counts(keep, rolled),
                rolls_left - 1,
            )
        return expected


def candidate_keeps(dice: Counts, used_mask: int) -> tuple[Counts, ...]:
    candidates: set[Counts] = {(0, 0, 0, 0, 0, 0), dice}

    for face_index, count in enumerate(dice):
        if count:
            keep = [0] * 6
            keep[face_index] = count
            candidates.add(tuple(keep))  # type: ignore[arg-type]

    faces = {face for face, count in enumerate(dice, start=1) if count}
    for run in ({1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}):
        keep = [0] * 6
        for face in run & faces:
            keep[face - 1] = 1
        if sum(keep) >= 2:
            candidates.add(tuple(keep))  # type: ignore[arg-type]

    # Pair plus kicker options matter for full house and high chance/kinds.
    for pair_face, pair_count in enumerate(dice, start=1):
        if pair_count >= 2:
            for kicker_face, kicker_count in enumerate(dice, start=1):
                keep = [0] * 6
                keep[pair_face - 1] = pair_count
                if kicker_face != pair_face and kicker_count:
                    keep[kicker_face - 1] = 1
                candidates.add(tuple(keep))  # type: ignore[arg-type]

    return tuple(keep for keep in candidates if can_keep(keep, dice))
