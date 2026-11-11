# Showcase copy of services/api/src/mahjong_api/playbyplay.py from the private mahjong-companion repo. Here to be read, not run.

"""A play-by-play of every game in Langfuse, next to the AI steps.

Each move becomes one Langfuse event in the game's session, named like the move ("Player 3
discards 9 bam"), with where it came from (tapped, auto-logged from a heard line, confirmed from a
card, or the call advice) and the player's position after it (top hand, tiles away, any call
advice). Fixes and undos are entries of their own, and Mahjong adds a game summary. The session
view then reads as a replay, with heard lines, strategist asks and photo readings in between.

Recording must never break the game: any Langfuse error is swallowed.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from langfuse import propagate_attributes

from mahjong_api.strategist import langfuse_client

log = logging.getLogger(__name__)
# Keys the app adds to an event to say where it came from; the game rules ignore them.
SOURCE_KEYS = ("source", "heard", "heard_trace")


def describe(event: dict[str, Any]) -> str:
    who = "You" if event.get("player") == 1 else f"Player {event.get('player')}"
    match event.get("type"):
        case "deal":
            return f"Dealt {len(event.get('tiles', []))} tiles"
        case "pass":
            return f"Charleston: passed {', '.join(event.get('out', []))}"
        case "draw":
            return f"You draw {event.get('tile')}"
        case "discard":
            return f"{who} discards {event.get('tile')}"
        case "call":
            tiles = event.get("tiles", [])
            return f"{who} calls and exposes {len(tiles)}x {tiles[0] if tiles else '?'}"
        case "swap":
            return f"{who} swaps for a joker"
        case "mahjong":
            return f"{who} declares Mahjong"
    return str(event.get("type"))


def record_move(game_id: str, index: int, event: dict[str, Any], view: dict[str, Any]) -> None:
    """One move, with where it came from and the player's position after it."""
    _record(
        game_id,
        f"{index + 1}. {describe(event)}",
        input={k: v for k, v in event.items() if k not in SOURCE_KEYS},
        output=_position(view),
        metadata={"move": index + 1, **_source(event)},
    )
    if event.get("type") == "mahjong":
        record_summary(game_id, view)


def record_fix(
    game_id: str, index: int, before: dict[str, Any], after: dict[str, Any], view: dict
) -> None:
    _record(
        game_id,
        f"Changed move {index + 1}: {describe(before)} -> {describe(after)}",
        input={"before": before, "after": after},
        output=_position(view),
        metadata={"move": index + 1, "source": "fix"},
    )


def record_undo(game_id: str, removed: dict[str, Any], view: dict[str, Any]) -> None:
    _record(
        game_id,
        f"Undid: {describe(removed)}",
        input=removed,
        output=_position(view),
        metadata={"source": "undo"},
    )


def record_summary(game_id: str, view: dict[str, Any]) -> None:
    events = view.get("events", [])
    sources = Counter(e.get("source", "tap") for e in events if e.get("type") == "discard")
    winner = view.get("winner")
    _record(
        game_id,
        f"Game over: {'you won' if winner == 1 else f'Player {winner} won'}",
        input={"players": view.get("players"), "cards": view.get("cards")},
        output={
            "winner": winner,
            "moves": len(events),
            "discards_by_player": {p: len(t) for p, t in view.get("discards", {}).items()},
            "how_discards_were_logged": dict(sources),
            "your_closest_hand": _position(view).get("top_hand"),
        },
        metadata={"source": "summary"},
    )
    try:
        langfuse_client().flush()  # the app may scale to zero soon after a game ends
    except Exception:
        log.warning("Langfuse flush failed", exc_info=True)


def _position(view: dict[str, Any]) -> dict[str, Any]:
    advice = view.get("advice") or {}
    hands = advice.get("hands") or []
    top = hands[0] if hands else None
    return {
        "turn": view.get("turn"),
        "phase": view.get("phase"),
        "top_hand": f"{top['label']} ({top['id']}), {top['needed']} away" if top else None,
        "call_advice": [f"{c['kind']} for {c['hand_id']}" for c in advice.get("calls") or []],
    }


def _source(event: dict[str, Any]) -> dict[str, Any]:
    source = {k: event[k] for k in SOURCE_KEYS if event.get(k)}
    source.setdefault("source", "tap")
    return source


def _record(game_id: str, name: str, **fields: Any) -> None:
    try:
        langfuse = langfuse_client()
        with propagate_attributes(session_id=game_id, trace_name=name, tags=["play-by-play"]):
            langfuse.create_event(name=name, **fields)
    except Exception:
        log.warning("Couldn't record a move in Langfuse", exc_info=True)
