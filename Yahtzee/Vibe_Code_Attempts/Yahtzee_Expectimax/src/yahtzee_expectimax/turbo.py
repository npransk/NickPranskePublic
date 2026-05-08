from __future__ import annotations

from yahtzee_expectimax.constants import CATEGORIES, UPPER_BONUS, UPPER_BONUS_TARGET, YAHTZEE_BONUS
from yahtzee_expectimax.dice import Counts
from yahtzee_expectimax.scoring import is_yahtzee, score_counts


def _open_categories(used_mask: int) -> list[str]:
    return [category for index, category in enumerate(CATEGORIES) if not used_mask & (1 << index)]


class TurboPolicy:
    """Very fast hand-tuned policy for high-volume simulation."""

    def choose(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts, rolls_left: int) -> tuple[str, Counts | str]:
        if rolls_left <= 0:
            return "score", self.score_category(used_mask, upper_total, yahtzee_bonus_enabled, dice)

        made_large = score_counts("large_straight", dice) == 40 and self._open(used_mask, "large_straight")
        made_small = score_counts("small_straight", dice) == 30 and self._open(used_mask, "small_straight")
        made_full = score_counts("full_house", dice) == 25 and self._open(used_mask, "full_house")
        made_yahtzee = is_yahtzee(dice) and self._open(used_mask, "yahtzee")
        if made_yahtzee or made_large or (rolls_left == 1 and (made_small or made_full)):
            return "score", self.score_category(used_mask, upper_total, yahtzee_bonus_enabled, dice)

        return "keep", self.keep_counts(used_mask, upper_total, dice)

    def keep_counts(self, used_mask: int, upper_total: int, dice: Counts) -> Counts:
        open_set = set(_open_categories(used_mask))
        max_count = max(dice)
        best_face = max(range(1, 7), key=lambda face: (dice[face - 1], face))

        if "yahtzee" in open_set and max_count >= 3:
            return self._only_face(dice, best_face)
        if "four_of_kind" in open_set and max_count >= 3:
            return self._only_face(dice, best_face)
        if "three_of_kind" in open_set and max_count >= 3:
            return self._only_face(dice, best_face)

        faces = {face for face, count in enumerate(dice, start=1) if count}
        if "large_straight" in open_set:
            keep = self._best_run_keep(dice, ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}))
            if sum(keep) >= 4:
                return keep
        if "small_straight" in open_set:
            keep = self._best_run_keep(dice, ({1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}))
            if sum(keep) >= 3:
                return keep

        upper_faces = [index + 1 for index, category in enumerate(CATEGORIES[:6]) if category in open_set]
        if upper_faces:
            pressure = UPPER_BONUS_TARGET - upper_total
            target_face = max(upper_faces, key=lambda face: (dice[face - 1] * face + face * (2 if pressure > 0 else 0), dice[face - 1], face))
            if dice[target_face - 1] >= 2 or target_face >= 4:
                return self._only_face(dice, target_face)

        if "full_house" in open_set and max_count >= 2:
            return self._only_face(dice, best_face)

        return self._only_face(dice, best_face)

    def score_category(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts) -> str:
        open_categories = _open_categories(used_mask)
        open_set = set(open_categories)
        scored = {category: score_counts(category, dice) for category in open_categories}

        if "yahtzee" in open_set and scored["yahtzee"] == 50:
            return "yahtzee"
        if "large_straight" in open_set and scored["large_straight"] == 40:
            return "large_straight"
        if "small_straight" in open_set and scored["small_straight"] == 30:
            return "small_straight"
        if "full_house" in open_set and scored["full_house"] == 25:
            return "full_house"

        upper_options = []
        for index, category in enumerate(CATEGORIES[:6]):
            if category in open_set:
                face = index + 1
                score = scored[category]
                target = face * 3
                value = score + (8 if score >= target else 0) + min(score, max(0, UPPER_BONUS_TARGET - upper_total)) * 1.3
                if upper_total + score >= UPPER_BONUS_TARGET:
                    value += UPPER_BONUS
                upper_options.append((value, category))

        lower_options = []
        for category in open_categories:
            if category not in CATEGORIES[:6]:
                value = scored[category]
                if category == "chance" and len(open_categories) > 4:
                    value -= 6
                if category == "four_of_kind" and value == 0 and len(open_categories) > 3:
                    value -= 8
                if category == "yahtzee" and value == 0 and len(open_categories) > 2:
                    value -= 16
                if category != "yahtzee" and yahtzee_bonus_enabled and is_yahtzee(dice):
                    value += YAHTZEE_BONUS
                lower_options.append((value, category))

        best_value, best_category = max(upper_options + lower_options)
        if best_value > 0:
            return best_category

        sacrifice_order = ["ones", "twos", "three_of_kind", "four_of_kind", "chance", "full_house", "small_straight", "large_straight", "yahtzee", "threes", "fours", "fives", "sixes"]
        return next(category for category in sacrifice_order if category in open_set)

    def _open(self, used_mask: int, category: str) -> bool:
        return not used_mask & (1 << CATEGORIES.index(category))

    def _only_face(self, dice: Counts, face: int) -> Counts:
        keep = [0] * 6
        keep[face - 1] = dice[face - 1]
        return tuple(keep)  # type: ignore[return-value]

    def _best_run_keep(self, dice: Counts, runs: tuple[set[int], ...]) -> Counts:
        faces = {face for face, count in enumerate(dice, start=1) if count}
        run = max(runs, key=lambda candidate: (len(candidate & faces), sum(candidate & faces)))
        keep = [0] * 6
        for face in run & faces:
            keep[face - 1] = 1
        return tuple(keep)  # type: ignore[return-value]
