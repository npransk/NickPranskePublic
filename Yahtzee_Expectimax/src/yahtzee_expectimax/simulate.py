from __future__ import annotations

from dataclasses import dataclass
from random import Random
import time

import numpy as np

from yahtzee_expectimax.dice import counts_to_dice
from yahtzee_expectimax.game import GameState
from yahtzee_expectimax.hybrid import HybridPolicy
from yahtzee_expectimax.solver import ExpectimaxSolver
from yahtzee_expectimax.tournament import TournamentPolicy
from yahtzee_expectimax.turbo import TurboPolicy


@dataclass(frozen=True)
class TurnLog:
    turn: int
    rolls: tuple[tuple[int, ...], ...]
    kept: tuple[tuple[int, ...], ...]
    category: str
    gained: int
    total: int


def play_game(solver: ExpectimaxSolver, seed: int | None = None, verbose: bool = False) -> tuple[int, list[TurnLog]]:
    game = GameState(rng=Random(seed))
    logs: list[TurnLog] = []

    for turn in range(1, 14):
        rolls: list[tuple[int, ...]] = []
        keeps: list[tuple[int, ...]] = []
        dice = game.roll_all()
        rolls.append(counts_to_dice(dice))
        rolls_left = 2

        while rolls_left > 0:
            decision = solver.best_action(game.used_mask, game.upper_total, game.yahtzee_bonus_enabled, game.dice, rolls_left)
            if decision.kind == "score":
                break
            keep = decision.payload
            keeps.append(counts_to_dice(keep))  # type: ignore[arg-type]
            dice = game.reroll(keep)  # type: ignore[arg-type]
            rolls.append(counts_to_dice(dice))
            rolls_left -= 1

        decision = solver.best_action(game.used_mask, game.upper_total, game.yahtzee_bonus_enabled, game.dice, 0)
        category = str(decision.payload)
        gained = game.score_category(category)
        logs.append(TurnLog(turn, tuple(rolls), tuple(keeps), category, gained, game.score))

        if verbose:
            print(f"turn={turn} rolls={rolls} keeps={keeps} score={category}:{gained} total={game.score}")

    return game.score, logs


def simulate_games(games: int, seed: int = 42, verbose_first: bool = False) -> dict[str, object]:
    solver = ExpectimaxSolver()
    started = time.perf_counter()
    scores: list[int] = []
    first_log: list[TurnLog] = []

    for index in range(games):
        score, log = play_game(solver, seed=seed + index, verbose=verbose_first and index == 0)
        scores.append(score)
        if index == 0:
            first_log = log
        if (index + 1) % 10 == 0:
            print(f"simulated={index + 1} mean={np.mean(scores):.2f}")

    elapsed = time.perf_counter() - started
    arr = np.array(scores)
    return {
        "games": games,
        "mean_score": float(arr.mean()),
        "median_score": float(np.median(arr)),
        "std_score": float(arr.std()),
        "min_score": int(arr.min()),
        "max_score": int(arr.max()),
        "over_200_rate": float(np.mean(arr >= 200)),
        "over_240_rate": float(np.mean(arr >= 240)),
        "over_250_rate": float(np.mean(arr >= 250)),
        "elapsed_seconds": elapsed,
        "cache_info": solver.cache_info(),
        "first_game": [turn.__dict__ for turn in first_log],
    }


def play_game_with_policy(policy: TournamentPolicy, seed: int | None = None, verbose: bool = False) -> tuple[int, list[TurnLog]]:
    game = GameState(rng=Random(seed))
    logs: list[TurnLog] = []
    for turn in range(1, 14):
        rolls: list[tuple[int, ...]] = []
        keeps: list[tuple[int, ...]] = []
        game.roll_all()
        rolls.append(counts_to_dice(game.dice))
        rolls_left = 2

        while rolls_left > 0:
            kind, payload = policy.choose(game.used_mask, game.upper_total, game.yahtzee_bonus_enabled, game.dice, rolls_left)
            if kind == "score":
                break
            keeps.append(counts_to_dice(payload))  # type: ignore[arg-type]
            game.reroll(payload)  # type: ignore[arg-type]
            rolls.append(counts_to_dice(game.dice))
            rolls_left -= 1

        category = policy.choose(game.used_mask, game.upper_total, game.yahtzee_bonus_enabled, game.dice, 0)[1]
        gained = game.score_category(str(category))
        logs.append(TurnLog(turn, tuple(rolls), tuple(keeps), str(category), gained, game.score))
        if verbose:
            print(f"turn={turn} rolls={rolls} keeps={keeps} score={category}:{gained} total={game.score}")
    return game.score, logs


def simulate_tournament(games: int, seed: int = 42, verbose_first: bool = False) -> dict[str, object]:
    policy = TournamentPolicy()
    started = time.perf_counter()
    scores: list[int] = []
    first_log: list[TurnLog] = []

    for index in range(games):
        score, log = play_game_with_policy(policy, seed=seed + index, verbose=verbose_first and index == 0)
        scores.append(score)
        if index == 0:
            first_log = log
        if (index + 1) % 100 == 0:
            print(f"simulated={index + 1} mean={np.mean(scores):.2f}")

    elapsed = time.perf_counter() - started
    arr = np.array(scores)
    return {
        "games": games,
        "mean_score": float(arr.mean()),
        "median_score": float(np.median(arr)),
        "std_score": float(arr.std()),
        "min_score": int(arr.min()),
        "max_score": int(arr.max()),
        "over_200_rate": float(np.mean(arr >= 200)),
        "over_240_rate": float(np.mean(arr >= 240)),
        "over_250_rate": float(np.mean(arr >= 250)),
        "elapsed_seconds": elapsed,
        "first_game": [turn.__dict__ for turn in first_log],
    }


def simulate_turbo(games: int, seed: int = 42, verbose_first: bool = False) -> dict[str, object]:
    policy = TurboPolicy()
    started = time.perf_counter()
    scores: list[int] = []
    first_log: list[TurnLog] = []

    for index in range(games):
        score, log = play_game_with_policy(policy, seed=seed + index, verbose=verbose_first and index == 0)  # type: ignore[arg-type]
        scores.append(score)
        if index == 0:
            first_log = log

    elapsed = time.perf_counter() - started
    arr = np.array(scores)
    return {
        "games": games,
        "mean_score": float(arr.mean()),
        "median_score": float(np.median(arr)),
        "std_score": float(arr.std()),
        "min_score": int(arr.min()),
        "max_score": int(arr.max()),
        "over_200_rate": float(np.mean(arr >= 200)),
        "over_240_rate": float(np.mean(arr >= 240)),
        "over_250_rate": float(np.mean(arr >= 250)),
        "elapsed_seconds": elapsed,
        "first_game": [turn.__dict__ for turn in first_log],
    }


def simulate_hybrid(games: int, seed: int = 42, rollouts: int = 16, verbose_first: bool = False) -> dict[str, object]:
    policy = HybridPolicy(rollouts=rollouts, seed=seed)
    started = time.perf_counter()
    scores: list[int] = []
    first_log: list[TurnLog] = []

    for index in range(games):
        score, log = play_game_with_policy(policy, seed=seed + index, verbose=verbose_first and index == 0)  # type: ignore[arg-type]
        scores.append(score)
        if index == 0:
            first_log = log
        if (index + 1) % 10 == 0:
            print(f"simulated={index + 1} mean={np.mean(scores):.2f}")

    elapsed = time.perf_counter() - started
    arr = np.array(scores)
    return {
        "games": games,
        "mean_score": float(arr.mean()),
        "median_score": float(np.median(arr)),
        "std_score": float(arr.std()),
        "min_score": int(arr.min()),
        "max_score": int(arr.max()),
        "over_200_rate": float(np.mean(arr >= 200)),
        "over_240_rate": float(np.mean(arr >= 240)),
        "over_250_rate": float(np.mean(arr >= 250)),
        "elapsed_seconds": elapsed,
        "rollouts": rollouts,
        "first_game": [turn.__dict__ for turn in first_log],
    }
