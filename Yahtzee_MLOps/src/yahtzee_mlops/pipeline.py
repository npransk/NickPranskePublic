"""Local continuous-training pipeline.

This is intentionally lightweight: gameplay events are stored locally, the
pipeline checks whether enough new play has happened, then runs an incremental
training job and registers a new model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yahtzee_mlops.agent import AgentConfig
from yahtzee_mlops.telemetry import TelemetryStore
from yahtzee_mlops.train import TrainConfig, train


@dataclass
class PipelineConfig:
    telemetry_db: str = "data/gameplay.sqlite3"
    model_dir: str = "models"
    min_new_events: int = 100
    episodes_per_run: int = 2_000
    eval_games: int = 100
    seed: int = 42


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    store = TelemetryStore(Path(config.telemetry_db))
    new_events = store.unused_event_count()
    if new_events < config.min_new_events:
        return {
            "status": "skipped",
            "reason": f"Only {new_events} new events; need {config.min_new_events}.",
            "telemetry": store.summary(),
        }

    metrics = train(
        TrainConfig(
            episodes=config.episodes_per_run,
            eval_games=config.eval_games,
            seed=config.seed,
            model_dir=config.model_dir,
        ),
        AgentConfig(seed=config.seed),
        resume_latest=True,
    )
    store.mark_all_used()
    return {"status": "trained", "new_events": new_events, "metrics": metrics, "telemetry": store.summary()}
