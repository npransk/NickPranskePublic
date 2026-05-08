from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from yahtzee_expectimax.constants import CATEGORIES
from yahtzee_expectimax.simulate import play_game_with_policy
from yahtzee_expectimax.tournament import TournamentPolicy


def diagnose_tournament(games: int, seed: int, output: Path | None = None) -> dict[str, object]:
    policy = TournamentPolicy()
    started = time.perf_counter()
    scores: list[int] = []
    category_scores = {category: [] for category in CATEGORIES}
    upper_bonus_count = 0

    for index in range(games):
        score, logs = play_game_with_policy(policy, seed=seed + index)
        scores.append(score)
        upper = 0
        for log in logs:
            category_scores[log.category].append(log.gained)
            if log.category in CATEGORIES[:6]:
                upper += log.gained
        if upper >= 63:
            upper_bonus_count += 1
        if (index + 1) % 25 == 0:
            print(f"diagnosed={index + 1} mean={np.mean(scores):.2f} upper_bonus={upper_bonus_count / (index + 1):.2%}")

    arr = np.array(scores)
    category_summary = {}
    for category, values in category_scores.items():
        vals = np.array(values, dtype=np.float64)
        category_summary[category] = {
            "mean": float(vals.mean()) if vals.size else 0.0,
            "zero_rate": float(np.mean(vals == 0)) if vals.size else 0.0,
            "count": int(vals.size),
        }

    result = {
        "games": games,
        "mean_score": float(arr.mean()),
        "median_score": float(np.median(arr)),
        "std_score": float(arr.std()),
        "min_score": int(arr.min()),
        "max_score": int(arr.max()),
        "over_240_rate": float(np.mean(arr >= 240)),
        "upper_bonus_rate": upper_bonus_count / games,
        "elapsed_seconds": time.perf_counter() - started,
        "category_summary": category_summary,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
