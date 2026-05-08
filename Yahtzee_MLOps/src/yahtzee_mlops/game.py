"""Pure Yahtzee environment used by training, tests, CLI, and Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np

from yahtzee_mlops.actions import Action, keep_count_actions
from yahtzee_mlops.constants import (
    CATEGORIES,
    MAX_ROLLS,
    NUM_DICE,
    NUM_FACES,
    UPPER_BONUS_POINTS,
    UPPER_BONUS_TARGET,
    UPPER_CATEGORIES,
    YAHTZEE_BONUS_POINTS,
)
from yahtzee_mlops.scoring import face_counts, is_yahtzee, score_category


@dataclass
class YahtzeeGame:
    rng: random.Random = field(default_factory=random.Random)
    dice: tuple[int, ...] = (0, 0, 0, 0, 0)
    scorecard: dict[str, int | None] = field(default_factory=lambda: {cat: None for cat in CATEGORIES})
    roll_count: int = 0
    turn: int = 0
    yahtzee_bonus_count: int = 0
    upper_bonus_awarded: bool = False

    @property
    def first_roll_of_turn(self) -> bool:
        return self.roll_count == 0

    @property
    def total_score(self) -> int:
        base = sum(score for score in self.scorecard.values() if score is not None)
        return base + (UPPER_BONUS_POINTS if self.upper_bonus_awarded else 0)

    def reset(self) -> np.ndarray:
        self.dice = (0, 0, 0, 0, 0)
        self.scorecard = {cat: None for cat in CATEGORIES}
        self.roll_count = 0
        self.turn = 0
        self.yahtzee_bonus_count = 0
        self.upper_bonus_awarded = False
        return self.state_vector()

    def roll(self, keep_counts: tuple[int, ...] | None = None) -> tuple[int, ...]:
        if self.roll_count >= MAX_ROLLS:
            raise ValueError("Cannot roll more than three times in a turn.")
        if keep_counts is None:
            keep_counts = (0,) * NUM_FACES
        if len(keep_counts) != NUM_FACES:
            raise ValueError("keep_counts must have one entry per die face.")

        current_counts = face_counts(self.dice) if not self.first_roll_of_turn else (0,) * NUM_FACES
        if any(keep < 0 or keep > current for keep, current in zip(keep_counts, current_counts)):
            raise ValueError(f"Cannot keep counts {keep_counts} from dice {self.dice}.")
        if sum(keep_counts) > NUM_DICE:
            raise ValueError("Cannot keep more than five dice.")

        kept: list[int] = []
        for face, count in enumerate(keep_counts, start=1):
            kept.extend([face] * count)

        rolled = [self.rng.randint(1, NUM_FACES) for _ in range(NUM_DICE - len(kept))]
        self.dice = tuple(sorted(kept + rolled))
        self.roll_count += 1
        return self.dice

    def score(self, category: str) -> int:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        if self.scorecard[category] is not None:
            raise ValueError(f"Category already used: {category}")
        if self.first_roll_of_turn:
            raise ValueError("Must roll before scoring.")

        score = score_category(category, self.dice)
        if is_yahtzee(self.dice) and self.scorecard["yahtzee"] and category != "yahtzee":
            score += YAHTZEE_BONUS_POINTS
            self.yahtzee_bonus_count += 1

        self.scorecard[category] = score
        self.turn += 1
        self.roll_count = 0
        self.dice = (0, 0, 0, 0, 0)

        upper_total = sum(self.scorecard[cat] or 0 for cat in UPPER_CATEGORIES)
        if not self.upper_bonus_awarded and upper_total >= UPPER_BONUS_TARGET:
            self.upper_bonus_awarded = True
        return score

    def is_game_over(self) -> bool:
        return all(score is not None for score in self.scorecard.values())

    def legal_actions(self) -> list[Action]:
        if self.is_game_over():
            return []
        if self.first_roll_of_turn:
            return [("roll", (0, 0, 0, 0, 0, 0))]

        actions: list[Action] = []
        if self.roll_count < MAX_ROLLS:
            counts = face_counts(self.dice)
            actions.extend(
                ("roll", keep)
                for keep in keep_count_actions()
                if all(kept <= available for kept, available in zip(keep, counts))
            )

        actions.extend(("score", cat) for cat in CATEGORIES if self.scorecard[cat] is None)
        return actions

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        before = self.total_score
        action_type, payload = action
        info: dict[str, object] = {"action": action}

        if action_type == "roll":
            self.roll(payload)  # type: ignore[arg-type]
            reward = -0.02
        elif action_type == "score":
            gained = self.score(payload)  # type: ignore[arg-type]
            reward = float(self.total_score - before)
            info["score_delta"] = gained
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        done = self.is_game_over()
        if done:
            reward += float(self.total_score) * 0.1
        return self.state_vector(), reward, done, info

    def state_vector(self) -> np.ndarray:
        dice_counts = np.array(face_counts(self.dice), dtype=np.float32) / NUM_DICE if not self.first_roll_of_turn else np.zeros(6, dtype=np.float32)
        filled = np.array([self.scorecard[cat] is not None for cat in CATEGORIES], dtype=np.float32)
        scores = np.array([(self.scorecard[cat] or 0) / 50.0 for cat in CATEGORIES], dtype=np.float32)
        potentials = np.array(
            [
                score_category(cat, self.dice) / 50.0 if not self.first_roll_of_turn and self.scorecard[cat] is None else 0.0
                for cat in CATEGORIES
            ],
            dtype=np.float32,
        )
        upper_total = sum(self.scorecard[cat] or 0 for cat in UPPER_CATEGORIES)
        meta = np.array(
            [
                self.roll_count / MAX_ROLLS,
                self.turn / len(CATEGORIES),
                min(upper_total / UPPER_BONUS_TARGET, 1.5),
                float(self.upper_bonus_awarded),
                self.total_score / 400.0,
                self.yahtzee_bonus_count / 3.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([dice_counts, filled, scores, potentials, meta])


STATE_SIZE = 6 + 13 + 13 + 13 + 6
