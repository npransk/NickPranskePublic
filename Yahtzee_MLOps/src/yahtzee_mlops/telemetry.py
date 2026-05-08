"""Gameplay telemetry for continuous improvement."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    final_score INTEGER,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    roll_count INTEGER NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    used_for_training INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(game_id) REFERENCES games(game_id)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelemetryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def start_game(self, player_id: str = "anonymous", model_version: str | None = None) -> str:
        game_id = str(uuid4())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO games(game_id, player_id, started_at, model_version) VALUES (?, ?, ?, ?)",
                (game_id, player_id, utc_now(), model_version),
            )
        return game_id

    def finish_game(self, game_id: str, final_score: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE games SET finished_at = ?, final_score = ? WHERE game_id = ?",
                (utc_now(), final_score, game_id),
            )

    def log_event(self, game_id: str, turn: int, roll_count: int, actor: str, event_type: str, payload: dict[str, object]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO events(event_id, game_id, turn, roll_count, actor, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), game_id, turn, roll_count, actor, event_type, json.dumps(payload), utc_now()),
            )

    def unused_event_count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM events WHERE used_for_training = 0").fetchone()[0])

    def mark_all_used(self) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE events SET used_for_training = 1 WHERE used_for_training = 0")

    def summary(self) -> dict[str, int | float | None]:
        with self.connect() as conn:
            games = int(conn.execute("SELECT COUNT(*) FROM games WHERE finished_at IS NOT NULL").fetchone()[0])
            events = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            avg_score = conn.execute("SELECT AVG(final_score) FROM games WHERE final_score IS NOT NULL").fetchone()[0]
        return {"finished_games": games, "events": events, "average_human_score": avg_score}
