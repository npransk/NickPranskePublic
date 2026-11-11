# Showcase copy of services/api/src/mahjong_api/strategist.py from the private mahjong-companion repo. Here to be read, not run.

"""The strategist: Claude picks one of the engine's closest hands and says why (Phase 4).

The engine does every count (ADR 1). Claude gets the top candidates, the table and two engine
tools for a closer look, then weighs what the engine can't: what opponents seem to collect, how
exposed a hand would be, which tiles are getting scarce. It must answer with ``submit_pick``,
naming one of the candidates, so it can never invent a hand.

One Langfuse trace per question (session = the game), with a generation per Claude call and a
span per tool call. The player's "follow it" tap becomes the ``recommendation_followed`` score.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import Any

from langfuse import Langfuse, get_client, propagate_attributes
from mahjong_engine.card import Card
from mahjong_engine.game import ME, GameState, advise
from mahjong_engine.odds import HandOdds, closest_variants
from mahjong_engine.tiles import TileKind

MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MAX_CALLS = 4
CANDIDATES = 5

SYSTEM = """You advise one player (Player 1, "you") in a game of American Mahjong played with \
the NMJL card. A rules engine has already counted everything: for each candidate hand on the \
card it gives how many tiles are still needed, the live "outs" that would help, and the closest \
target. Trust its numbers; your job is the judgment it can't make.

Weigh: how close each hand is; whether opponents' discards and exposures suggest they want the \
same tiles; whether a hand is concealed (no calling allowed) or exposed; how many outs remain; \
and joker swaps the engine found. Use the tools when a closer look would change your pick.

Finish by calling submit_pick with one of the candidate hand IDs and a reason of one sentence, \
at most 25 words, written to the player ("Go for ... because ..."). Mention tiles by name."""

TOOLS = [
    {
        "name": "hand_details",
        "description": (
            "The closest ways to complete one candidate hand: for each variant, the target "
            "tiles, how many are still needed, the outs and any joker swaps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"hand_id": {"type": "string", "description": "e.g. 2024-ALN-2"}},
            "required": ["hand_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "opponent_reads",
        "description": (
            "Each opponent's discards and exposures, with discards counted by suit and "
            "honors, to judge what they are collecting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit_pick",
        "description": "Your final answer: the hand to go for and a one-sentence reason.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hand_id": {"type": "string", "description": "One of the candidate hand IDs"},
                "reason": {"type": "string", "description": "One sentence, at most 25 words"},
            },
            "required": ["hand_id", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@cache
def langfuse_client() -> Langfuse:
    """The Langfuse client for this app's project.

    ``LANGFUSE_PUBLIC_KEY_MJ`` / ``LANGFUSE_SECRET_KEY_MJ`` win when set, so a machine whose
    plain ``LANGFUSE_*`` keys belong to another project (Claude Code tracing) still traces here.
    In Azure the plain names come from Key Vault.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY_MJ")
    if not public_key:
        return get_client()
    Langfuse(
        public_key=public_key,
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY_MJ"),
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
    )
    return get_client(public_key=public_key)


class StrategistError(RuntimeError):
    """The strategist couldn't give an answer."""


@dataclass(frozen=True)
class Pick:
    hand_id: str
    label: str
    reason: str
    trace_id: str | None


def ask(client: Any, card: Card, state: GameState, game_id: str, reads: dict | None = None) -> Pick:
    """Ask Claude which candidate hand to pursue. ``client`` is an ``anthropic.Anthropic``."""
    candidates = advise(card, state, top=CANDIDATES).hands
    if not candidates:
        raise StrategistError("No hand on the card is still possible")
    by_id = {c.hand.id: c for c in candidates}
    langfuse = langfuse_client()

    with propagate_attributes(session_id=game_id, trace_name="strategist"):
        with langfuse.start_as_current_observation(
            name="strategist", as_type="agent", input={"candidates": list(by_id)}
        ) as agent:
            trace_id = langfuse.get_current_trace_id()
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": _situation(state, candidates)}
            ]
            for _ in range(MAX_CALLS):
                response = _call_claude(client, langfuse, messages)
                if response.stop_reason == "refusal":
                    raise StrategistError("The model declined to answer")
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if block.name == "submit_pick" and block.input.get("hand_id") in by_id:
                        pick = Pick(
                            block.input["hand_id"],
                            by_id[block.input["hand_id"]].hand.label,
                            block.input["reason"],
                            trace_id,
                        )
                        agent.update(output={"hand_id": pick.hand_id, "reason": pick.reason})
                        langfuse.flush()
                        return pick
                    results.append(_run_tool(langfuse, block, card, state, by_id, reads or {}))
                if not results:
                    # Answered in text instead of calling the tool: ask once more.
                    results = [{"type": "text", "text": "Call submit_pick with your choice."}]
                messages.append({"role": "user", "content": results})
            agent.update(level="ERROR", status_message="No pick after the call limit")
    langfuse.flush()
    raise StrategistError("The strategist didn't settle on a hand")


def _call_claude(client: Any, langfuse: Any, messages: list[dict[str, Any]]) -> Any:
    with langfuse.start_as_current_observation(
        name="claude",
        as_type="generation",
        model=MODEL,
        model_parameters={"effort": EFFORT},
        input=_jsonable(messages),
    ) as generation:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},
            # If a safety classifier ever declines, retry on Anthropic's recommended model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = response.usage
        generation.update(
            output=_jsonable(response.content),
            usage_details={
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            },
        )
        return response


def _run_tool(
    langfuse: Any, block: Any, card: Card, state: GameState, by_id: dict, reads: dict
) -> dict:
    with langfuse.start_as_current_observation(
        name=block.name, as_type="tool", input=_jsonable(block.input)
    ) as span:
        if block.name == "hand_details" and block.input.get("hand_id") in by_id:
            output = _hand_details(card, state, by_id[block.input["hand_id"]])
        elif block.name == "opponent_reads":
            output = _opponent_reads(state, reads)
        else:
            output = {"error": f"Unknown tool or hand: pick one of {sorted(by_id)}"}
        span.update(output=output)
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": json.dumps(output),
        **({"is_error": True} if "error" in output else {}),
    }


def _odds(odds: HandOdds) -> dict[str, Any]:
    return {
        "hand_id": odds.hand.id,
        "label": odds.hand.label,
        "concealed": odds.hand.concealed,
        "note": odds.hand.note,
        "tiles_needed": odds.needed,
        "outs": odds.outs,
        "joker_swaps": odds.swaps,
        "target": str(odds.variant),
    }


def _situation(state: GameState, candidates: list[HandOdds]) -> str:
    mine = state.exposures[ME]
    lines = [
        f"Players: {state.players}. Your tiles ({state.my_tiles.total()}): "
        + ", ".join(str(t) for t in sorted(state.concealed.elements())),
        "Your exposures: " + ("; ".join(_exposure(e) for e in mine) or "none"),
        f"Discards so far: {len(state.all_discards)}.",
        "Candidate hands from the engine, closest first:",
        json.dumps([_odds(c) for c in candidates], indent=1),
        "Which one should I go for?",
    ]
    return "\n".join(lines)


def _hand_details(card: Card, state: GameState, odds: HandOdds) -> dict[str, Any]:
    variants = closest_variants(
        odds.hand, state.my_tiles, state.all_discards, state.others_exposures, state.exposures[ME]
    )
    return {"hand": odds.hand.id, "closest_variants": [_odds(v) for v in variants[:5]]}


def _opponent_reads(state: GameState, reads: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for player in range(2, state.players + 1):
        discards = state.discards[player]
        by_kind = Counter(
            t.kind.value if t.kind in (TileKind.BAM, TileKind.CRAK, TileKind.DOT) else "honors"
            for t in discards
        )
        out[f"Player {player}"] = {
            "discards": [str(t) for t in discards],
            "discards_by_suit": dict(by_kind),
            "exposures": [_exposure(e) for e in state.exposures[player]],
        }
        if read := reads.get(player):
            # The engine's read of their own card: hands that fit their exposures and discards.
            out[f"Player {player}"]["likely_hands"] = (
                "too early to tell" if read.early else [x.hand.label for x in read.likely]
            )
            out[f"Player {player}"]["tiles_you_hold_they_may_need"] = [str(t) for t in read.careful]
    return out


def _exposure(e) -> str:
    jokers = f" ({e.jokers} joker{'s' if e.jokers > 1 else ''})" if e.jokers else ""
    return f"{e.size}x {e.tile}{jokers}"


def _jsonable(value: Any) -> Any:
    """SDK objects and plain data as JSON-friendly values for Langfuse."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value
