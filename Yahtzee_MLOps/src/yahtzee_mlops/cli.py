"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yahtzee_mlops.pipeline import PipelineConfig, run_pipeline
from yahtzee_mlops.telemetry import TelemetryStore
from yahtzee_mlops.agent import AgentConfig
from yahtzee_mlops.train import TrainConfig, benchmark_baselines, imitate_heuristic, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="yahtzee-mlops")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--episodes", type=int, default=10_000)
    train_parser.add_argument("--eval-games", type=int, default=200)
    train_parser.add_argument("--model-dir", default="models")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--fresh", action="store_true", help="Do not resume from latest model.")
    train_parser.add_argument("--train-every-steps", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--min-replay-size", type=int, default=5000)
    train_parser.add_argument("--hidden-size", type=int, default=256)
    train_parser.add_argument("--epsilon-decay", type=float, default=0.99995)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--games", type=int, default=1000)
    benchmark_parser.add_argument("--seed", type=int, default=42)

    imitate_parser = subparsers.add_parser("imitate")
    imitate_parser.add_argument("--episodes", type=int, default=2000)
    imitate_parser.add_argument("--epochs", type=int, default=3)
    imitate_parser.add_argument("--eval-games", type=int, default=200)
    imitate_parser.add_argument("--model-dir", default="models")
    imitate_parser.add_argument("--seed", type=int, default=42)
    imitate_parser.add_argument("--batch-size", type=int, default=256)
    imitate_parser.add_argument("--hidden-size", type=int, default=128)

    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument("--telemetry-db", default="data/gameplay.sqlite3")
    pipeline_parser.add_argument("--model-dir", default="models")
    pipeline_parser.add_argument("--min-new-events", type=int, default=100)
    pipeline_parser.add_argument("--episodes-per-run", type=int, default=2_000)
    pipeline_parser.add_argument("--eval-games", type=int, default=100)
    pipeline_parser.add_argument("--seed", type=int, default=42)

    telemetry_parser = subparsers.add_parser("telemetry")
    telemetry_parser.add_argument("--telemetry-db", default="data/gameplay.sqlite3")

    args = parser.parse_args()
    if args.command == "train":
        result = train(
            TrainConfig(
                episodes=args.episodes,
                eval_games=args.eval_games,
                model_dir=args.model_dir,
                seed=args.seed,
                train_every_steps=args.train_every_steps,
            ),
            AgentConfig(
                seed=args.seed,
                batch_size=args.batch_size,
                min_replay_size=args.min_replay_size,
                hidden_size=args.hidden_size,
                epsilon_decay=args.epsilon_decay,
            ),
            resume_latest=not args.fresh,
        )
    elif args.command == "pipeline":
        result = run_pipeline(
            PipelineConfig(
                telemetry_db=args.telemetry_db,
                model_dir=args.model_dir,
                min_new_events=args.min_new_events,
                episodes_per_run=args.episodes_per_run,
                eval_games=args.eval_games,
                seed=args.seed,
            )
        )
    elif args.command == "benchmark":
        result = benchmark_baselines(games=args.games, seed=args.seed)
    elif args.command == "imitate":
        result = imitate_heuristic(
            episodes=args.episodes,
            epochs=args.epochs,
            eval_games=args.eval_games,
            model_dir=args.model_dir,
            seed=args.seed,
            agent_config=AgentConfig(
                seed=args.seed,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                epsilon_start=0.0,
                epsilon_min=0.0,
            ),
        )
    else:
        result = TelemetryStore(Path(args.telemetry_db)).summary()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
