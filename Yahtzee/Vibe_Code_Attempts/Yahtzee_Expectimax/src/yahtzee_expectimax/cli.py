from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yahtzee_expectimax.numba_solver import random_search, refine_search, simulate_numba
from yahtzee_expectimax.diagnostics import diagnose_tournament
from yahtzee_expectimax.simulate import simulate_games, simulate_hybrid, simulate_tournament, simulate_turbo
from yahtzee_expectimax.solver import ExpectimaxSolver


def main() -> None:
    parser = argparse.ArgumentParser(prog="yahtzee-expectimax")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ev_parser = subparsers.add_parser("ev")
    ev_parser.add_argument("--upper", type=int, default=0)
    ev_parser.add_argument("--used-mask", type=int, default=0)
    ev_parser.add_argument("--yahtzee-bonus", action="store_true")

    sim_parser = subparsers.add_parser("simulate")
    sim_parser.add_argument("--games", type=int, default=100)
    sim_parser.add_argument("--seed", type=int, default=42)
    sim_parser.add_argument("--output", type=Path)
    sim_parser.add_argument("--verbose-first", action="store_true")

    fast_parser = subparsers.add_parser("tournament")
    fast_parser.add_argument("--games", type=int, default=1000)
    fast_parser.add_argument("--seed", type=int, default=42)
    fast_parser.add_argument("--output", type=Path)
    fast_parser.add_argument("--verbose-first", action="store_true")

    turbo_parser = subparsers.add_parser("turbo")
    turbo_parser.add_argument("--games", type=int, default=10000)
    turbo_parser.add_argument("--seed", type=int, default=42)
    turbo_parser.add_argument("--output", type=Path)
    turbo_parser.add_argument("--verbose-first", action="store_true")

    hybrid_parser = subparsers.add_parser("hybrid")
    hybrid_parser.add_argument("--games", type=int, default=100)
    hybrid_parser.add_argument("--seed", type=int, default=42)
    hybrid_parser.add_argument("--rollouts", type=int, default=16)
    hybrid_parser.add_argument("--output", type=Path)
    hybrid_parser.add_argument("--verbose-first", action="store_true")

    numba_parser = subparsers.add_parser("numba")
    numba_parser.add_argument("--games", type=int, default=1000)
    numba_parser.add_argument("--seed", type=int, default=42)
    numba_parser.add_argument("--output", type=Path)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--iterations", type=int, default=50)
    search_parser.add_argument("--games", type=int, default=500)
    search_parser.add_argument("--seed", type=int, default=42)
    search_parser.add_argument("--output", type=Path)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--params-file", type=Path, required=True)
    validate_parser.add_argument("--games", type=int, default=5000)
    validate_parser.add_argument("--seed", type=int, default=90210)
    validate_parser.add_argument("--output", type=Path)

    refine_parser = subparsers.add_parser("refine")
    refine_parser.add_argument("--source", type=Path, required=True)
    refine_parser.add_argument("--iterations", type=int, default=50)
    refine_parser.add_argument("--games", type=int, default=3000)
    refine_parser.add_argument("--seed", type=int, default=42)
    refine_parser.add_argument("--output", type=Path)

    diag_parser = subparsers.add_parser("diagnose")
    diag_parser.add_argument("--games", type=int, default=100)
    diag_parser.add_argument("--seed", type=int, default=42)
    diag_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "ev":
        solver = ExpectimaxSolver()
        result = {
            "expected_value": solver._game_value(args.used_mask, min(args.upper, 63), args.yahtzee_bonus),
            "cache_info": solver.cache_info(),
        }
    elif args.command == "simulate":
        result = simulate_games(args.games, seed=args.seed, verbose_first=args.verbose_first)
    elif args.command == "tournament":
        result = simulate_tournament(args.games, seed=args.seed, verbose_first=args.verbose_first)
    elif args.command == "turbo":
        result = simulate_turbo(args.games, seed=args.seed, verbose_first=args.verbose_first)
    elif args.command == "numba":
        result = simulate_numba(args.games, seed=args.seed)
    elif args.command == "search":
        result = random_search(args.iterations, args.games, args.seed, args.output)
    elif args.command == "validate":
        payload = json.loads(args.params_file.read_text(encoding="utf-8"))
        params = np.array(payload["best"]["params"], dtype=np.float64)
        result = simulate_numba(args.games, args.seed, params=params)
    elif args.command == "refine":
        result = refine_search(args.source, args.iterations, args.games, args.seed, args.output)
    elif args.command == "diagnose":
        result = diagnose_tournament(args.games, args.seed, args.output)
    else:
        result = simulate_hybrid(args.games, seed=args.seed, rollouts=args.rollouts, verbose_first=args.verbose_first)

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
