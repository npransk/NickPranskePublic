from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
from numba import njit

from yahtzee_expectimax.constants import CATEGORIES
from yahtzee_expectimax.dice import ALL_DICE_COUNTS, ALL_KEEP_COUNTS, Counts, add_counts, can_keep, roll_distribution
from yahtzee_expectimax.scoring import score_counts
from yahtzee_expectimax.tournament import candidate_keeps


@dataclass(frozen=True)
class NumbaTables:
    counts: np.ndarray
    all_keeps: np.ndarray
    score_table: np.ndarray
    legal_keeps: np.ndarray
    legal_keep_indices: np.ndarray
    legal_keep_counts: np.ndarray
    transitions: np.ndarray
    transition_probs: np.ndarray
    transition_counts: np.ndarray


def build_tables() -> NumbaTables:
    counts = np.array(ALL_DICE_COUNTS, dtype=np.int16)
    all_keeps = np.array(ALL_KEEP_COUNTS, dtype=np.int16)
    index = {counts_tuple: i for i, counts_tuple in enumerate(ALL_DICE_COUNTS)}
    score_table = np.zeros((len(ALL_DICE_COUNTS), len(CATEGORIES)), dtype=np.int16)
    for dice_index, dice in enumerate(ALL_DICE_COUNTS):
        for category_index, category in enumerate(CATEGORIES):
            score_table[dice_index, category_index] = score_counts(category, dice)

    max_keeps = len(ALL_KEEP_COUNTS)
    legal_keeps_array = np.full((len(ALL_DICE_COUNTS), max_keeps, 6), -1, dtype=np.int16)
    legal_keep_indices = np.full((len(ALL_DICE_COUNTS), max_keeps), -1, dtype=np.int16)
    legal_keep_counts = np.zeros(len(ALL_DICE_COUNTS), dtype=np.int16)
    keep_lookup = {keep: i for i, keep in enumerate(ALL_KEEP_COUNTS)}
    for dice_index, dice in enumerate(ALL_DICE_COUNTS):
        keep_index = 0
        for keep in candidate_keeps(dice, 0):
            if can_keep(keep, dice):
                legal_keeps_array[dice_index, keep_index] = keep
                legal_keep_indices[dice_index, keep_index] = keep_lookup[keep]
                keep_index += 1
        legal_keep_counts[dice_index] = keep_index

    max_outcomes = max(len(roll_distribution(n)) for n in range(6))
    transitions = np.full((len(ALL_KEEP_COUNTS), max_outcomes), -1, dtype=np.int16)
    transition_probs = np.zeros((len(ALL_KEEP_COUNTS), max_outcomes), dtype=np.float64)
    transition_counts = np.zeros(len(ALL_KEEP_COUNTS), dtype=np.int16)
    for keep in ALL_KEEP_COUNTS:
        keep_index = keep_lookup[keep]
        outcomes = roll_distribution(5 - sum(keep))
        transition_counts[keep_index] = len(outcomes)
        for outcome_index, (rolled, probability) in enumerate(outcomes):
            transitions[keep_index, outcome_index] = index[add_counts(keep, rolled)]
            transition_probs[keep_index, outcome_index] = probability

    return NumbaTables(counts, all_keeps, score_table, legal_keeps_array, legal_keep_indices, legal_keep_counts, transitions, transition_probs, transition_counts)


TABLES = build_tables()
KEEP_INDEX = {keep: i for i, keep in enumerate(ALL_KEEP_COUNTS)}
COUNT_INDEX = {counts: i for i, counts in enumerate(ALL_DICE_COUNTS)}


@njit(cache=True)
def _score_utility(score: int, category: int, used_mask: int, upper_total: int, ybonus: int, dice_counts: np.ndarray, params: np.ndarray) -> float:
    value = float(score)
    open_count = 13
    mask = used_mask
    while mask:
        open_count -= mask & 1
        mask >>= 1

    if category < 6:
        target = (category + 1) * 3
        deficit = max(0, 63 - upper_total)
        value += min(score, deficit) * params[0]
        if score >= target:
            value += params[1] + (category + 1) * params[2]
        else:
            value -= (target - score) * params[3]
        if upper_total < 63 and upper_total + score >= 63:
            value += 35.0
    else:
        baseline = 0.0
        if category == 6:
            baseline = params[4]
        elif category == 7:
            baseline = params[5]
        elif category == 8:
            baseline = params[6]
        elif category == 9:
            baseline = params[7]
        elif category == 10:
            baseline = params[8]
        elif category == 11:
            baseline = params[9]
        elif category == 12:
            baseline = params[10]
        value += score - baseline

    if category == 12 and open_count > 4:
        value -= params[11]
    if category == 11:
        if score == 50:
            value += params[12]
        elif open_count > 2:
            value -= params[13]
    if category == 10:
        if score == 40:
            value += params[14]
        elif open_count > 4:
            value -= params[15]

    max_count = 0
    for i in range(6):
        if dice_counts[i] > max_count:
            max_count = dice_counts[i]
    if category != 11 and ybonus == 1 and max_count == 5:
        value += 100.0
    return value


@njit(cache=True)
def _best_score(score_table: np.ndarray, counts: np.ndarray, dice_index: int, used_mask: int, upper_total: int, ybonus: int, params: np.ndarray) -> tuple[int, float]:
    best_category = -1
    best_value = -1e18
    for category in range(13):
        if used_mask & (1 << category):
            continue
        score = int(score_table[dice_index, category])
        value = _score_utility(score, category, used_mask, upper_total, ybonus, counts[dice_index], params)
        if value > best_value:
            best_value = value
            best_category = category
    return best_category, best_value


@njit(cache=True)
def _turn_value(score_table: np.ndarray, counts: np.ndarray, legal_keep_indices: np.ndarray, legal_keep_counts: np.ndarray, transitions: np.ndarray, transition_probs: np.ndarray, transition_counts: np.ndarray, dice_index: int, used_mask: int, upper_total: int, ybonus: int, rolls_left: int, params: np.ndarray) -> float:
    best_category, best_value = _best_score(score_table, counts, dice_index, used_mask, upper_total, ybonus, params)
    if rolls_left <= 0:
        return best_value

    keep_count = int(legal_keep_counts[dice_index])
    for local_keep_index in range(keep_count):
        keep_index = int(legal_keep_indices[dice_index, local_keep_index])
        expected = 0.0
        outcomes = int(transition_counts[keep_index])
        for outcome_index in range(outcomes):
            next_dice = int(transitions[keep_index, outcome_index])
            expected += transition_probs[keep_index, outcome_index] * _turn_value(
                score_table,
                counts,
                legal_keep_indices,
                legal_keep_counts,
                transitions,
                transition_probs,
                transition_counts,
                next_dice,
                used_mask,
                upper_total,
                ybonus,
                rolls_left - 1,
                params,
            )
        if expected > best_value:
            best_value = expected
    return best_value


@njit(cache=True)
def _choose_action(score_table: np.ndarray, counts: np.ndarray, legal_keep_indices: np.ndarray, legal_keep_counts: np.ndarray, transitions: np.ndarray, transition_probs: np.ndarray, transition_counts: np.ndarray, dice_index: int, used_mask: int, upper_total: int, ybonus: int, rolls_left: int, params: np.ndarray) -> tuple[int, int]:
    best_category, best_value = _best_score(score_table, counts, dice_index, used_mask, upper_total, ybonus, params)
    best_kind = 1
    best_payload = best_category
    if rolls_left <= 0:
        return best_kind, best_payload

    keep_count = int(legal_keep_counts[dice_index])
    for local_keep_index in range(keep_count):
        keep_index = int(legal_keep_indices[dice_index, local_keep_index])
        expected = 0.0
        outcomes = int(transition_counts[keep_index])
        for outcome_index in range(outcomes):
            next_dice = int(transitions[keep_index, outcome_index])
            expected += transition_probs[keep_index, outcome_index] * _turn_value(
                score_table,
                counts,
                legal_keep_indices,
                legal_keep_counts,
                transitions,
                transition_probs,
                transition_counts,
                next_dice,
                used_mask,
                upper_total,
                ybonus,
                rolls_left - 1,
                params,
            )
        if expected > best_value - params[16]:
            best_value = expected
            best_kind = 0
            best_payload = keep_index
    return best_kind, best_payload


@njit(cache=True)
def _compute_turn_values(score_table: np.ndarray, counts: np.ndarray, legal_keep_indices: np.ndarray, legal_keep_counts: np.ndarray, transitions: np.ndarray, transition_probs: np.ndarray, transition_counts: np.ndarray, used_mask: int, upper_total: int, ybonus: int, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((3, counts.shape[0]), dtype=np.float64)
    actions = np.zeros((3, counts.shape[0], 2), dtype=np.int16)

    for dice_index in range(counts.shape[0]):
        category, value = _best_score(score_table, counts, dice_index, used_mask, upper_total, ybonus, params)
        values[0, dice_index] = value
        actions[0, dice_index, 0] = 1
        actions[0, dice_index, 1] = category

    for rolls_left in range(1, 3):
        prev = rolls_left - 1
        for dice_index in range(counts.shape[0]):
            category, best_value = _best_score(score_table, counts, dice_index, used_mask, upper_total, ybonus, params)
            best_kind = 1
            best_payload = category
            keep_count = int(legal_keep_counts[dice_index])
            for local_keep_index in range(keep_count):
                keep_index = int(legal_keep_indices[dice_index, local_keep_index])
                expected = 0.0
                outcomes = int(transition_counts[keep_index])
                for outcome_index in range(outcomes):
                    next_dice = int(transitions[keep_index, outcome_index])
                    expected += transition_probs[keep_index, outcome_index] * values[prev, next_dice]
                if expected > best_value - params[16]:
                    best_value = expected
                    best_kind = 0
                    best_payload = keep_index
            values[rolls_left, dice_index] = best_value
            actions[rolls_left, dice_index, 0] = best_kind
            actions[rolls_left, dice_index, 1] = best_payload
    return values, actions, values[0]


@njit(cache=True)
def _roll_index_from_random(counts: np.ndarray) -> int:
    rolled = np.zeros(6, dtype=np.int16)
    for _ in range(5):
        rolled[np.random.randint(0, 6)] += 1
    for i in range(counts.shape[0]):
        equal = True
        for j in range(6):
            if counts[i, j] != rolled[j]:
                equal = False
                break
        if equal:
            return i
    return 0


@njit(cache=True)
def _roll_from_keep_index(counts: np.ndarray, keep: np.ndarray) -> int:
    rolled = np.zeros(6, dtype=np.int16)
    for i in range(6):
        rolled[i] = keep[i]
    to_roll = 5
    for i in range(6):
        to_roll -= keep[i]
    for _ in range(to_roll):
        rolled[np.random.randint(0, 6)] += 1
    for i in range(counts.shape[0]):
        equal = True
        for j in range(6):
            if counts[i, j] != rolled[j]:
                equal = False
                break
        if equal:
            return i
    return 0


@njit(cache=True)
def _play_games(score_table: np.ndarray, counts: np.ndarray, all_keeps: np.ndarray, legal_keep_indices: np.ndarray, legal_keep_counts: np.ndarray, transitions: np.ndarray, transition_probs: np.ndarray, transition_counts: np.ndarray, params: np.ndarray, games: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    scores = np.zeros(games, dtype=np.int32)
    for game_index in range(games):
        used_mask = 0
        upper_total = 0
        ybonus = 0
        total_score = 0
        for _turn in range(13):
            dice_index = _roll_index_from_random(counts)
            rolls_left = 2
            _values, actions, _score_values = _compute_turn_values(
                score_table,
                counts,
                legal_keep_indices,
                legal_keep_counts,
                transitions,
                transition_probs,
                transition_counts,
                used_mask,
                upper_total,
                ybonus,
                params,
            )
            while rolls_left > 0:
                kind = int(actions[rolls_left, dice_index, 0])
                payload = int(actions[rolls_left, dice_index, 1])
                if kind == 1:
                    break
                dice_index = _roll_from_keep_index(counts, all_keeps[payload])
                rolls_left -= 1
            category, _value = _best_score(score_table, counts, dice_index, used_mask, upper_total, ybonus, params)
            score = int(score_table[dice_index, category])
            max_count = 0
            for i in range(6):
                if counts[dice_index, i] > max_count:
                    max_count = counts[dice_index, i]
            if category != 11 and ybonus == 1 and max_count == 5:
                score += 100
            if category < 6:
                old_upper = upper_total
                upper_total += int(score_table[dice_index, category])
                if upper_total > 63:
                    upper_total = 63
                if old_upper < 63 and upper_total >= 63:
                    score += 35
            if category == 11 and int(score_table[dice_index, category]) == 50:
                ybonus = 1
            used_mask |= 1 << category
            total_score += score
        scores[game_index] = total_score
    return scores


DEFAULT_PARAMS = np.array(
    [
        0.78,  # upper progress
        4.0,  # upper target bonus
        0.35,  # face bonus
        0.90,  # upper miss penalty
        21.0,
        13.0,
        22.0,
        24.0,
        18.0,
        12.0,
        23.0,
        8.0,  # early chance penalty
        35.0,  # made yahtzee bonus
        18.0,  # zero yahtzee penalty
        7.0,  # made large straight bonus
        5.0,  # zero large straight penalty
        2.0,  # roll-vs-score margin
    ],
    dtype=np.float64,
)


def simulate_numba(games: int, seed: int, params: np.ndarray | None = None) -> dict[str, object]:
    params = DEFAULT_PARAMS.copy() if params is None else params.astype(np.float64)
    started = time.perf_counter()
    scores = _play_games(
        TABLES.score_table,
        TABLES.counts,
        TABLES.all_keeps,
        TABLES.legal_keep_indices,
        TABLES.legal_keep_counts,
        TABLES.transitions,
        TABLES.transition_probs,
        TABLES.transition_counts,
        params,
        games,
        seed,
    )
    elapsed = time.perf_counter() - started
    return summarize_scores(scores, elapsed, params)


def summarize_scores(scores: np.ndarray, elapsed: float, params: np.ndarray) -> dict[str, object]:
    return {
        "games": int(scores.size),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "std_score": float(scores.std()),
        "min_score": int(scores.min()),
        "max_score": int(scores.max()),
        "over_200_rate": float(np.mean(scores >= 200)),
        "over_240_rate": float(np.mean(scores >= 240)),
        "over_250_rate": float(np.mean(scores >= 250)),
        "elapsed_seconds": elapsed,
        "params": [float(x) for x in params],
    }


def random_search(iterations: int, games: int, seed: int, output: Path | None = None) -> dict[str, object]:
    rng = random.Random(seed)
    best_params = DEFAULT_PARAMS.copy()
    best = simulate_numba(games, seed, best_params)
    history = [best]
    for iteration in range(iterations):
        candidate = best_params.copy()
        for index in range(candidate.size):
            scale = 0.15 if index < 4 else 0.25
            candidate[index] *= rng.uniform(1.0 - scale, 1.0 + scale)
        result = simulate_numba(games, seed + 1000 + iteration, candidate)
        history.append(result)
        if result["mean_score"] > best["mean_score"]:
            best = result
            best_params = candidate
            print(f"iteration={iteration} new_best={best['mean_score']:.3f}")
        elif iteration % 10 == 0:
            print(f"iteration={iteration} candidate={result['mean_score']:.3f} best={best['mean_score']:.3f}")

    payload = {"best": best, "history": history}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def refine_search(source: Path, iterations: int, games: int, seed: int, output: Path | None = None) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    best_params = np.array(payload["best"]["params"], dtype=np.float64)
    best = simulate_numba(games, seed, best_params)
    history = [best]
    for iteration in range(iterations):
        candidate = best_params.copy()
        for index in range(candidate.size):
            candidate[index] *= rng.uniform(0.92, 1.08)
        result = simulate_numba(games, seed + 2000 + iteration, candidate)
        history.append(result)
        if result["mean_score"] > best["mean_score"]:
            best = result
            best_params = candidate
            print(f"iteration={iteration} new_best={best['mean_score']:.3f}")
        elif iteration % 10 == 0:
            print(f"iteration={iteration} candidate={result['mean_score']:.3f} best={best['mean_score']:.3f}")
    refined = {"best": best, "history": history, "source": str(source)}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(refined, indent=2), encoding="utf-8")
    return refined
