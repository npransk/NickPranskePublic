# Showcase copy of services/api/src/mahjong_api/listen.py from the private mahjong-companion repo. Here to be read, not run.

"""From a heard line to a proposed game event, traced so the player's answer can score it.

The parser (``mahjong_engine.heard``) says what was said; the game state says who said it: a
discard belongs to the player whose turn it is (ADR 3). A call or Mahjong comes back without a
player, for the confirm card to ask. Each line is one Langfuse trace (session = the game); the
player's confirm, fix or ignore becomes the ``discard_parse_correct`` score.

Lines the parser isn't sure of go to Jev (ADR 4): is it a discard announcement, and which tile?
Clean announcements skip it. If Jev fails, the parser's reading stands.
"""

from __future__ import annotations

from typing import Any

from langfuse import propagate_attributes
from mahjong_engine.game import GameState
from mahjong_engine.heard import Heard, parse_line
from mahjong_engine.names import parse_tile

from mahjong_api import jev
from mahjong_api.strategist import langfuse_client

# "auto": logged without asking because it was 90%+ sure; a later change rescores it "fixed".
OUTCOMES = {"confirmed": 1.0, "auto": 1.0, "fixed": 0.0, "ignored": 0.0}
CONFIDENT = 0.9  # the parser's clean announcements skip Jev
# Jev guesses below this are noise, not proposals ("MMM." came back as 3 dot, 13% sure).
FLOOR = 0.3


def propose(state: GameState, heard: Heard) -> list[dict[str, Any]]:
    """The events the line most likely means, in order; empty for chatter or a line that doesn't
    fit. Several discards in one line ("Red Dragon. 8 crack.") go to successive players."""
    if state.phase != "play":
        return []
    if heard.kind == "discard" and heard.tiles:
        player, events = state.turn, []
        for tile in heard.tiles:
            events.append({"type": "discard", "player": player, "tile": str(tile)})
            player = state.next_player(player)
        return events
    if heard.kind == "call" and state.last_discard is not None:
        return [{"type": "call", "player": None, "tile": str(state.last_discard[1])}]
    if heard.kind == "mahjong":
        return [{"type": "mahjong", "player": None}]
    return []


def hear(
    state: GameState, text: str, game_id: str, jev_client: Any | None = None
) -> dict[str, Any]:
    """Parse a line, ask Jev when the parser isn't sure, and propose the event."""
    langfuse = langfuse_client()
    with propagate_attributes(session_id=game_id, trace_name="heard"):
        with langfuse.start_as_current_observation(
            name="heard", as_type="chain", input={"text": text, "turn": state.turn}
        ) as chain:
            trace_id = langfuse.get_current_trace_id()
            with langfuse.start_as_current_observation(name="parse-line", as_type="span") as span:
                heard = parse_line(text)
                span.update(output=_summary(heard))
            decided_by = "grammar"
            if jev_client is not None and _unsure(heard):
                judged = _ask_jev(langfuse, jev_client, state, heard)
                # If Jev fails, the parser's reading stands.
                decided_by = "jev" if judged is not heard else "grammar"
                heard = judged
            proposals = propose(state, heard)
            chain.update(
                output={**_summary(heard), "proposals": proposals},
                metadata={"decided_by": decided_by},
            )
    langfuse.flush()
    return {
        "text": text,
        "kind": heard.kind,
        "tile": str(heard.tile) if heard.tile else None,
        "confidence": heard.confidence,
        "decided_by": decided_by,
        "proposals": proposals,
        "proposal": proposals[0] if proposals else None,
        "trace_id": trace_id,
    }


def _unsure(heard: Heard) -> bool:
    """Calls and Mahjong are clear words; clean announcements need no second opinion."""
    return heard.kind in ("discard", "chatter") and heard.confidence < CONFIDENT


def _ask_jev(langfuse: Any, client: Any, state: GameState, heard: Heard) -> Heard:
    turn = "your" if state.turn == 1 else f"Player {state.turn}'s"
    with langfuse.start_as_current_observation(
        name="jev", as_type="generation", model=jev.MODEL, input={"line": heard.text}
    ) as generation:
        try:
            verdict = jev.judge(client, heard.text, turn)
        except Exception as e:  # Jev being down must not lose the line: keep the parser's view
            generation.update(level="ERROR", status_message=str(e))
            return heard
        generation.update(
            model=verdict.model,
            output={
                "game_talk": verdict.game_talk,
                "tile": verdict.tile,
                "p_tile": verdict.tile_probability,
            },
            usage_details={"input": verdict.input_tokens, "output": verdict.output_tokens},
            cost_details={"input": verdict.cost, "total": verdict.cost},
        )
    confidence = round(verdict.game_talk * verdict.tile_probability, 2)
    if verdict.game_talk < jev.GAME_TALK or confidence < FLOOR:
        return Heard("chatter", None, 0.0, heard.text)
    return Heard("discard", parse_tile(verdict.tile), confidence, heard.text)


def _summary(heard: Heard) -> dict[str, Any]:
    tile = str(heard.tile) if heard.tile else None
    return {"kind": heard.kind, "tile": tile, "confidence": heard.confidence}


def score(trace_id: str, outcome: str) -> float:
    """Record whether the line was read right. One score per line, overwritten on a later answer
    (an auto-logged discard changed afterwards goes from "auto" to "fixed")."""
    value = OUTCOMES[outcome]
    langfuse = langfuse_client()
    langfuse.create_score(
        score_id=f"{trace_id}-discard-parse",
        name="discard_parse_correct",
        value=value,
        data_type="NUMERIC",
        trace_id=trace_id,
        comment=outcome,
    )
    langfuse.flush()
    return value
