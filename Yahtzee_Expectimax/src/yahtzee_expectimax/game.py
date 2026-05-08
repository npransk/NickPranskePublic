from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from yahtzee_expectimax.constants import CATEGORIES, UPPER_BONUS, UPPER_BONUS_TARGET, YAHTZEE_BONUS
from yahtzee_expectimax.dice import Counts, reroll_from_keep, roll_all
from yahtzee_expectimax.scoring import is_yahtzee, score_counts


@dataclass
class GameState:
    rng: Random = field(default_factory=Random)
    used_mask: int = 0
    upper_total: int = 0
    score: int = 0
    yahtzee_bonus_enabled: bool = False
    dice: Counts = (0, 0, 0, 0, 0, 0)

    def roll_all(self) -> Counts:
        self.dice = roll_all(self.rng)
        return self.dice

    def reroll(self, keep: Counts) -> Counts:
        self.dice = reroll_from_keep(keep, self.rng)
        return self.dice

    def score_category(self, category: str) -> int:
        index = CATEGORIES.index(category)
        if self.used_mask & (1 << index):
            raise ValueError(f"Category already used: {category}")

        gained = score_counts(category, self.dice)
        if category != "yahtzee" and self.yahtzee_bonus_enabled and is_yahtzee(self.dice):
            gained += YAHTZEE_BONUS

        if index < 6:
            old_upper = self.upper_total
            self.upper_total = min(UPPER_BONUS_TARGET, self.upper_total + gained)
            if old_upper < UPPER_BONUS_TARGET <= self.upper_total:
                gained += UPPER_BONUS

        if category == "yahtzee" and score_counts(category, self.dice) == 50:
            self.yahtzee_bonus_enabled = True

        self.used_mask |= 1 << index
        self.score += gained
        return gained

    def is_done(self) -> bool:
        return self.used_mask == (1 << len(CATEGORIES)) - 1
