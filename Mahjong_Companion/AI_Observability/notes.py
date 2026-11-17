# Showcase copy of services/api/src/mahjong_api/notes.py from the private mahjong-companion repo. Here to be read, not run.

"""What the AI said about a game, kept beside it: the strategist's picks and the game review.

The game file holds only the moves, which every tap replays, so these live in a file of their
own in the same place, ``games/<id>.ai.json``:

    {"asks": [{"ask": 1, "move": 12, "hand_id": "2024-ALN-2", "label": "...", "reason": "...",
               "trace_id": "...", "followed": true}],
     "review": {...}}

``move`` is how many moves had been played when the strategist was asked. ``followed`` is the
player's "I'll go for it" (true) or "Not this one" (false) tap, or null. Strategist asks from
before this file existed are only in Langfuse.
"""

from __future__ import annotations

import json
from typing import Any

from mahjong_api.store import Store

SUFFIX = ".ai.json"


def name(game_id: str) -> str:
    return f"games/{game_id}{SUFFIX}"


def is_notes(stored: str) -> bool:
    """True for a notes file in the games folder, which isn't a game."""
    return stored.endswith(SUFFIX)


def load(store: Store, game_id: str) -> dict[str, Any]:
    text = store.get(name(game_id))
    notes = json.loads(text) if text else {}
    notes.setdefault("asks", [])
    notes.setdefault("review", None)
    return notes


def save(store: Store, game_id: str, notes: dict[str, Any]) -> None:
    store.put(name(game_id), json.dumps(notes, indent=1))


def add_ask(
    store: Store,
    game_id: str,
    move: int,
    hand_id: str,
    label: str,
    reason: str,
    trace_id: str | None,
) -> int:
    """Keep one strategist pick; returns its number in the game, from 1."""
    notes = load(store, game_id)
    number = len(notes["asks"]) + 1
    notes["asks"].append(
        {
            "ask": number,
            "move": move,
            "hand_id": hand_id,
            "label": label,
            "reason": reason,
            "trace_id": trace_id,
            "followed": None,
        }
    )
    save(store, game_id, notes)
    return number


def mark_followed(
    store: Store, game_id: str, followed: bool, *, ask: int | None, trace_id: str | None
) -> bool:
    """Record the player's tap on a pick, found by its number or its trace. False when the pick
    isn't kept (asked before picks were kept with the game)."""
    notes = load(store, game_id)
    for entry in notes["asks"]:
        if (ask is not None and entry["ask"] == ask) or (
            ask is None and trace_id and entry["trace_id"] == trace_id
        ):
            entry["followed"] = followed
            save(store, game_id, notes)
            return True
    return False
