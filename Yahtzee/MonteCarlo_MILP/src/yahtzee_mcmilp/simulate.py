"""Play full games and summarise scores in the same format as the repo's other results."""

from __future__ import annotations

import time

import numpy as np

from yahtzee_mcmilp.agent import Agent
from yahtzee_mcmilp.rules import (
    CATEGORIES, HANDS, KEEPS, N_CATEGORIES, State, faces_from_counts, hand_index,
)


def play_game(agent: Agent, rng: np.random.Generator, log: list | None = None) -> int:
    state = State()
    total = 0
    for _ in range(N_CATEGORIES):
        dice = list(rng.integers(1, 7, size=5))
        hand = hand_index(dice)
        for rolls_left in (2, 1):
            keep = KEEPS[agent.choose_keep(state, hand, rolls_left)]
            if keep.sum() == 5:
                break
            dice = faces_from_counts(keep) + list(rng.integers(1, 7, size=5 - int(keep.sum())))
            hand = hand_index(dice)
        category = agent.choose_category(state, hand)
        points, state = state.apply(category, hand)
        total += points
        if log is not None:
            log.append((faces_from_counts(HANDS[hand]), CATEGORIES[category], points))
    return total


def summarize(scores: list[int], elapsed: float) -> dict[str, float]:
    arr = np.array(scores)
    return {
        "games": len(arr),
        "mean": round(float(arr.mean()), 2),
        "stderr": round(float(arr.std(ddof=1) / np.sqrt(len(arr))), 2) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "std": round(float(arr.std()), 2),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "over_200": round(float((arr >= 200).mean()), 4),
        "over_250": round(float((arr >= 250).mean()), 4),
        "seconds": round(elapsed, 1),
    }


def run_games(agent: Agent, games: int, seed: int = 2024, progress: bool = False) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    start = time.time()
    for g in range(games):
        scores.append(play_game(agent, rng))
        if progress and (g + 1) % max(1, games // 10) == 0:
            print(f"  {g + 1}/{games} games, running mean {np.mean(scores):.1f}", flush=True)
    return summarize(scores, time.time() - start)
