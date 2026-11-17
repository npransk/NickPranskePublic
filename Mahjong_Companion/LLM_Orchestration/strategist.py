# Showcase copy of services/api/src/mahjong_api/strategist.py from the private mahjong-companion repo. Here to be read, not run.

"""The strategist: Claude picks one of the engine's closest hands and says why (Phase 4).

The engine does every count (ADR 1). Claude gets the top candidates, the table and two engine
tools for a closer look, then weighs what the engine can't: what opponents seem to collect, how
exposed a hand would be, which tiles are getting scarce. It must answer with ``submit_pick``,
naming one of the candidates, so it can never invent a hand.

One Langfuse trace per question (session = the game): an agent observation that holds the whole
situation, a generation per Claude call and a tool observation per tool call (ADR 7). Checks that
need no judgment are scored as soon as the answer is in: where the pick stood in the engine's
ranking, whether the reason kept to its word limit, how many calls it took, how the ask ended.
The player's "follow it" tap becomes ``recommendation_followed``, and once the game is over the
game review grades each pick with hindsight (``strategist_hindsight``, ``mahjong_api.review``).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from mahjong_engine.card import Card
from mahjong_engine.game import ME, GameState, advise
from mahjong_engine.odds import HandOdds, closest_variants
from mahjong_engine.tiles import TileKind

from mahjong_api.observability import (
    flush,
    langfuse_client,
    marking_errors,
    prompt_link,
    prompt_version,
    record_claude,
    record_score,
    trace,
)

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MAX_CALLS = 4
MAX_TOKENS = 4096
CANDIDATES = 5
REASON_WORDS = 25  # the limit the prompt sets; scored as reason_within_limit
PROMPT_NAME = "strategist"

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


# Changes whenever the instructions or the tools do; on every strategist trace.
PROMPT_VERSION = prompt_version(SYSTEM, TOOLS)


class StrategistError(RuntimeError):
    """The strategist couldn't give an answer."""


@dataclass(frozen=True)
class Pick:
    hand_id: str
    label: str
    reason: str
    trace_id: str | None
    # What Claude looked up on the way (name, input, output), shown with each regression run.
    tools: tuple[dict[str, Any], ...] = ()


@dataclass
class _Tally:
    """What one ask did, for its trace and scores. The outcome stays "error" unless the loop
    reaches one of its own endings."""

    llm_calls: int = 0
    tool_calls: int = 0
    rejected_picks: int = 0  # submit_pick naming a hand outside the candidates
    nudges: int = 0  # answered in text instead of a tool call
    outcome: str = "error"  # picked, refused, no_pick, error


def ask(client: Any, card: Card, state: GameState, game_id: str, reads: dict | None = None) -> Pick:
    """Ask Claude which candidate hand to pursue. ``client`` is an ``anthropic.Anthropic``."""
    candidates = advise(card, state, top=CANDIDATES).hands
    if not candidates:
        raise StrategistError("No hand on the card is still possible")
    by_id = {c.hand.id: c for c in candidates}
    langfuse = langfuse_client()
    situation = _situation(state, candidates)
    tally, pick, trace_id = _Tally(), None, None
    try:
        with (
            trace(
                "strategist",
                game_id=game_id,
                feature="strategist",
                version=PROMPT_VERSION,
                prompt=prompt_link(PROMPT_NAME, PROMPT_VERSION),
                card=card.year,
                players=state.players,
                discards=len(state.all_discards),
            ),
            langfuse.start_as_current_observation(
                name="strategist",
                as_type="agent",
                # Everything the answer was based on, so it can be judged from this alone.
                input={"situation": situation, "candidates": [_odds(c) for c in candidates]},
            ) as agent,
            marking_errors(agent),
        ):
            trace_id = langfuse.get_current_trace_id()
            try:
                pick = _decide(client, langfuse, card, state, by_id, reads or {}, situation, tally)
            finally:
                agent.update(metadata=asdict(tally))
            agent.update(
                output={
                    "hand_id": pick.hand_id,
                    "label": pick.label,
                    "reason": pick.reason,
                    "engine_rank": list(by_id).index(pick.hand_id) + 1,
                }
            )
    finally:
        _score(langfuse, trace_id, tally, pick, list(by_id))
        flush()
    return pick


def _decide(
    client: Any,
    langfuse: Any,
    card: Card,
    state: GameState,
    by_id: dict[str, HandOdds],
    reads: dict,
    situation: str,
    tally: _Tally,
) -> Pick:
    """The tool loop: call Claude until it submits one of the candidates."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": situation}]
    looked_up: list[dict[str, Any]] = []
    for round_number in range(1, MAX_CALLS + 1):
        tally.llm_calls += 1
        response = _call_claude(client, langfuse, messages, round_number)
        if response.stop_reason == "refusal":
            tally.outcome = "refused"
            raise StrategistError("The model declined to answer")
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "submit_pick":
                if block.input.get("hand_id") in by_id:
                    tally.outcome = "picked"
                    return Pick(
                        block.input["hand_id"],
                        by_id[block.input["hand_id"]].hand.label,
                        block.input["reason"],
                        langfuse.get_current_trace_id(),
                        tuple(looked_up),
                    )
                tally.rejected_picks += 1
            else:
                tally.tool_calls += 1
            output = _run_tool(langfuse, block, card, state, by_id, reads)
            if block.name != "submit_pick" and "error" not in output:
                looked_up.append({"name": block.name, "input": block.input, "output": output})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output),
                    **({"is_error": True} if "error" in output else {}),
                }
            )
        if not results:
            # Answered in text instead of calling the tool: ask once more.
            tally.nudges += 1
            results = [{"type": "text", "text": "Call submit_pick with your choice."}]
        messages.append({"role": "user", "content": results})
    tally.outcome = "no_pick"
    raise StrategistError("The strategist didn't settle on a hand")


def _score(
    langfuse: Any, trace_id: str | None, tally: _Tally, pick: Pick | None, ranking: list[str]
) -> None:
    """The checks that need no judgment, one score each per ask. Never raises."""
    if not trace_id:
        return

    def put(name: str, value: float | bool | str, comment: str | None = None) -> None:
        record_score(
            langfuse,
            name,
            value,
            trace_id=trace_id,
            score_id=f"{trace_id}-{name}",
            comment=comment,
        )

    try:
        put("strategist_outcome", tally.outcome)
        put("strategist_llm_calls", tally.llm_calls)
        put("strategist_tool_calls", tally.tool_calls)
        put("strategist_rejected_picks", tally.rejected_picks)
        if pick is not None:
            words = len(pick.reason.split())
            put("pick_engine_rank", ranking.index(pick.hand_id) + 1, comment=pick.hand_id)
            put("reason_words", words)
            put("reason_within_limit", words <= REASON_WORDS, comment=f"{words} words")
    except Exception:
        log.warning("Couldn't score a strategist ask", exc_info=True)


def _call_claude(client: Any, langfuse: Any, messages: list[dict[str, Any]], round_number: int):
    with (
        langfuse.start_as_current_observation(
            name="claude",
            as_type="generation",
            model=MODEL,
            model_parameters={"effort": EFFORT, "max_tokens": MAX_TOKENS},
            input=[{"role": "system", "content": SYSTEM}, *_jsonable(messages)],
            metadata={"round": round_number, "tools": [tool["name"] for tool in TOOLS]},
        ) as generation,
        marking_errors(generation),
    ):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},
            # If a safety classifier ever declines, retry on Anthropic's recommended model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        record_claude(generation, response, MODEL, _jsonable(response.content))
        return response


def _run_tool(
    langfuse: Any, block: Any, card: Card, state: GameState, by_id: dict, reads: dict
) -> dict[str, Any]:
    """Run one tool call and return what it found (or an "error" for Claude to act on)."""
    with langfuse.start_as_current_observation(
        name=block.name, as_type="tool", input=_jsonable(block.input)
    ) as span:
        if block.name == "hand_details" and block.input.get("hand_id") in by_id:
            output = _hand_details(card, state, by_id[block.input["hand_id"]])
        elif block.name == "opponent_reads":
            output = _opponent_reads(state, reads)
        else:
            output = {"error": f"Unknown tool or hand: pick one of {sorted(by_id)}"}
        span.update(output=output, level="WARNING" if "error" in output else None)
    return output


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
