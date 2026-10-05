"""Command line entry point.

    python -m yahtzee_mcmilp.cli simulate --agent milp --games 200 --workers 4
    python -m yahtzee_mcmilp.cli play --seed 7        # one logged game
    python -m yahtzee_mcmilp.cli value                # MILP estimate of a fresh scorecard
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from yahtzee_mcmilp.agent import GreedyAgent, MilpAgent
from yahtzee_mcmilp.milp import MilpValueEstimator
from yahtzee_mcmilp.rules import State
from yahtzee_mcmilp.scenarios import ScenarioBank
from yahtzee_mcmilp.simulate import play_game, summarize


def make_agent(args):
    if args.agent == "greedy":
        return GreedyAgent()
    bank = ScenarioBank(args.scenarios, seed=args.bank_seed)
    return MilpAgent(MilpValueEstimator(bank), discount=args.discount)


def _worker(job):
    args, seed, games = job
    agent = make_agent(args)
    rng = np.random.default_rng(seed)
    return [play_game(agent, rng) for _ in range(games)]


def cmd_simulate(args) -> None:
    import time

    per = [args.games // args.workers + (i < args.games % args.workers) for i in range(args.workers)]
    jobs = [(args, args.seed + 1000 * i, n) for i, n in enumerate(per) if n]
    start = time.time()
    with ProcessPoolExecutor(len(jobs)) as pool:
        scores = [s for chunk in pool.map(_worker, jobs) for s in chunk]
    result = summarize(scores, time.time() - start)
    result["config"] = {k: v for k, v in vars(args).items() if k not in ("func", "output")}
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps({**result, "scores": scores}, indent=2))


def cmd_play(args) -> None:
    log: list = []
    total = play_game(make_agent(args), np.random.default_rng(args.seed), log)
    for turn, (dice, category, points) in enumerate(log, 1):
        print(f"turn {turn:2d}: {dice} -> {category:15s} +{points}")
    print(f"total: {total}")


def cmd_value(args) -> None:
    est = MilpValueEstimator(ScenarioBank(args.scenarios, seed=args.bank_seed))
    print(f"MILP hindsight value of a fresh game: {est.value(State()):.1f} "
          f"(true optimum under these rules is about 254)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="yahtzee-mcmilp")
    sub = parser.add_subparsers(required=True)
    for name, func in (("simulate", cmd_simulate), ("play", cmd_play), ("value", cmd_value)):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--agent", choices=("milp", "greedy"), default="milp")
        p.add_argument("--scenarios", type=int, default=64, help="Monte Carlo scenarios per state")
        p.add_argument("--bank-seed", type=int, default=0)
        p.add_argument("--discount", type=float, default=1.0, help="multiplier on MILP future value")
        p.add_argument("--seed", type=int, default=2024)
        if name == "simulate":
            p.add_argument("--games", type=int, default=100)
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
