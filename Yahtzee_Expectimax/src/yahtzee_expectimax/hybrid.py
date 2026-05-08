from __future__ import annotations

from random import Random

from yahtzee_expectimax.constants import CATEGORIES, UPPER_BONUS, UPPER_BONUS_TARGET, YAHTZEE_BONUS
from yahtzee_expectimax.dice import Counts, reroll_from_keep
from yahtzee_expectimax.game import GameState
from yahtzee_expectimax.scoring import is_yahtzee, score_counts
from yahtzee_expectimax.tournament import TournamentPolicy, candidate_keeps


class HybridPolicy:
    """Tournament policy with Monte Carlo adjudication for close decisions.

    This is intentionally compute-hungry. It uses the fast tournament value to
    narrow candidates, then simulates continuations when the best roll/score
    choice is close or when selecting among score categories.
    """

    def __init__(self, rollouts: int = 64, seed: int = 12345) -> None:
        self.base = TournamentPolicy()
        self.rollouts = rollouts
        self.seed = seed

    def choose(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts, rolls_left: int) -> tuple[str, Counts | str]:
        if rolls_left <= 0:
            return "score", self._best_score_by_rollout(used_mask, upper_total, yahtzee_bonus_enabled, dice)

        base_kind, base_payload = self.base.choose(used_mask, upper_total, yahtzee_bonus_enabled, dice, rolls_left)
        score_category, score_value = self.base.best_score_category(used_mask, upper_total, yahtzee_bonus_enabled, dice)
        keep, keep_value = self.base.best_keep(used_mask, upper_total, yahtzee_bonus_enabled, dice, rolls_left)

        if abs(score_value - keep_value) > 6.0:
            return base_kind, base_payload

        score_rollout = self._rollout_score_action(used_mask, upper_total, yahtzee_bonus_enabled, dice, score_category)
        keep_rollout = self._rollout_keep_action(used_mask, upper_total, yahtzee_bonus_enabled, keep, rolls_left)
        if score_rollout >= keep_rollout:
            return "score", score_category
        return "keep", keep

    def _best_score_by_rollout(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts) -> str:
        candidates = []
        for index, category in enumerate(CATEGORIES):
            if used_mask & (1 << index):
                continue
            utility = self.base.best_score_category(used_mask, upper_total, yahtzee_bonus_enabled, dice)[1]
            raw = score_counts(category, dice)
            candidates.append((raw + utility, category))
        shortlist = [category for _, category in sorted(candidates, reverse=True)[:4]]
        return max(
            shortlist,
            key=lambda category: self._rollout_score_action(used_mask, upper_total, yahtzee_bonus_enabled, dice, category),
        )

    def _rollout_score_action(
        self,
        used_mask: int,
        upper_total: int,
        yahtzee_bonus_enabled: bool,
        dice: Counts,
        category: str,
    ) -> float:
        game = GameState(Random(0), used_mask, upper_total, 0, yahtzee_bonus_enabled, dice)
        immediate = game.score_category(category)
        return immediate + self._rollout_future(game.used_mask, game.upper_total, game.yahtzee_bonus_enabled, self._action_seed(used_mask, dice, category))

    def _rollout_keep_action(
        self,
        used_mask: int,
        upper_total: int,
        yahtzee_bonus_enabled: bool,
        keep: Counts,
        rolls_left: int,
    ) -> float:
        rng = Random(self._action_seed(used_mask, keep, str(rolls_left)))
        total = 0.0
        for _ in range(self.rollouts):
            dice = keep
            for _roll in range(rolls_left):
                dice = reroll_from_keep(dice, rng)
            category = self._best_score_by_rollout_light(used_mask, upper_total, yahtzee_bonus_enabled, dice)
            total += self._rollout_score_action(used_mask, upper_total, yahtzee_bonus_enabled, dice, category)
        return total / self.rollouts

    def _rollout_future(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, seed: int) -> float:
        if used_mask == (1 << len(CATEGORIES)) - 1:
            return 0.0
        total = 0.0
        for offset in range(self.rollouts):
            rng = Random(seed + offset)
            game = GameState(rng, used_mask, upper_total, 0, yahtzee_bonus_enabled)
            while not game.is_done():
                game.roll_all()
                rolls_left = 2
                while rolls_left > 0:
                    kind, payload = self.base.choose(game.used_mask, game.upper_total, game.yahtzee_bonus_enabled, game.dice, rolls_left)
                    if kind == "score":
                        break
                    game.reroll(payload)  # type: ignore[arg-type]
                    rolls_left -= 1
                category = self._best_score_by_rollout_light(game.used_mask, game.upper_total, game.yahtzee_bonus_enabled, game.dice)
                game.score_category(category)
            total += game.score
        return total / self.rollouts

    def _best_score_by_rollout_light(self, used_mask: int, upper_total: int, yahtzee_bonus_enabled: bool, dice: Counts) -> str:
        best_category = ""
        best_value = float("-inf")
        for index, category in enumerate(CATEGORIES):
            if used_mask & (1 << index):
                continue
            raw = score_counts(category, dice)
            value = raw
            if index < 6:
                target = (index + 1) * 3
                if raw >= target:
                    value += 8
                if upper_total + raw >= UPPER_BONUS_TARGET:
                    value += UPPER_BONUS
                value += min(raw, max(0, UPPER_BONUS_TARGET - upper_total)) * 0.8
            if category == "yahtzee":
                value += 25 if raw == 50 else -15
            if category == "large_straight":
                value += 8 if raw == 40 else -3
            if category != "yahtzee" and yahtzee_bonus_enabled and is_yahtzee(dice):
                value += YAHTZEE_BONUS
            if value > best_value:
                best_category = category
                best_value = value
        return best_category

    def _action_seed(self, used_mask: int, dice: Counts, label: str) -> int:
        return abs(hash((self.seed, used_mask, dice, label))) % 2_000_000_000
