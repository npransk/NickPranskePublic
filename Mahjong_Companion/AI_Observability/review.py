# Showcase copy of services/api/src/mahjong_api/review.py from the private mahjong-companion repo. Here to be read, not run.

"""Score My Game: Claude looks back over a finished game and coaches the player (ADR 8).

The engine replays the game and works out the facts (``mahjong_engine.hindsight``): at each of
your passes, discards and chances to call, what the engine would have done and what your choice
changed; any Mahjong you had and didn't declare; and how each strategist pick fared afterwards.
Claude is given those facts and nothing else, and writes the coaching: a 1-10 score for your
play, what went well, the moments that mattered most, a verdict on each strategist pick and one
thing to work on. The engine counts, the AI judges (ADR 1).

Claude Opus 5.5 writes it. It is a stronger model than the strategist it grades (Sonnet 5.5), so
the strategist isn't marking its own work, and it runs once a game, at about $0.10-0.20. The
answer has a fixed shape (a JSON schema); code then drops any moment or verdict that points at
a move or an ask that doesn't exist, and counts what it dropped.

In Langfuse (ADR 7) each review is one trace in the game's session: an evaluator observation
holding the engine's replay and the Claude call. Its scores: ``review_play_score`` on the game,
``strategist_hindsight`` on each strategist pick's own trace (so the strategist's dashboards
show how its picks held up), ``review_outcome`` and ``review_dropped_refs`` on the review, and
the player's "Useful?" tap as ``review_helpful``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from mahjong_engine.card import Card
from mahjong_engine.game import Game
from mahjong_engine.hindsight import Pick, facts

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

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000  # thinking included
PROMPT_NAME = "review-game"
MOMENTS = 5
WENT_WELL = 3
VERDICTS = ("good", "reasonable", "poor")

SYSTEM = """You coach one player (Player 1, "you") after a game of American Mahjong played \
with the NMJL card. A rules engine has replayed the game and lists the facts: at each \
Charleston pass and each of the player's discards, what the engine would have done, what the \
player did, and how many tiles they were from their closest hand afterwards either way; the \
discards they could have called and what they did; any Mahjong they had and didn't declare; and, \
for each time they asked the strategist (another AI) which hand to go for, what it said and how \
that hand fared afterwards. Moves are numbered from 1, as the app's log numbers them.

The engine's counts are exact; your job is the judgment. American Mahjong has a lot of luck, so \
score the decisions, not the result: a well-played game can be lost to the draw. The engine is \
a guide, not an oracle. Its discard advice looks only at the player's own hands, so a \
different discard can be right, for example keeping a tile an opponent seems to need. Say so \
when that is the case.

play_score: 1 to 10 for the player's decisions. 5 is a steady game with a few slips; 8 or more \
means few or no costly choices; 3 or less means several moves that clearly cost tiles, or a \
missed Mahjong.

moments: the 3 to 5 moves that mattered most, good or bad, in game order. For a good move, \
leave "better" empty.

strategist: one verdict per strategist ask. Judge what could be known when it was asked; use \
what happened later only to explain. good: a sound pick, whether or not it worked out. \
reasonable: defensible, but a better choice was visible then. poor: another hand was clearly \
better on what was known at the time.

Write to the player in plain words, a sentence or two per point. Name tiles and moves. Only \
cite moves, tiles and hands that appear in the facts."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "play_score": {"type": "integer", "enum": list(range(1, 11))},
        "summary": {"type": "string", "description": "Two or three sentences on the game."},
        "went_well": {"type": "array", "items": {"type": "string"}},
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "move": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["good", "mistake", "missed chance"]},
                    "what_happened": {"type": "string"},
                    "better": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["move", "kind", "what_happened", "better", "why"],
                "additionalProperties": False,
            },
        },
        "strategist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ask": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "why": {"type": "string"},
                },
                "required": ["ask", "verdict", "why"],
                "additionalProperties": False,
            },
        },
        "strategist_summary": {
            "type": "string",
            "description": "One or two sentences on the strategist's advice; empty if never asked.",
        },
        "work_on": {"type": "string", "description": "One thing to work on next game."},
    },
    "required": [
        "play_score",
        "summary",
        "went_well",
        "moments",
        "strategist",
        "strategist_summary",
        "work_on",
    ],
    "additionalProperties": False,
}

# Changes whenever the instructions or the answer shape do; on every review trace.
PROMPT_VERSION = prompt_version(SYSTEM, SCHEMA)


class ReviewError(RuntimeError):
    """The game couldn't be reviewed."""


class Refused(ReviewError):
    """Claude, and the fallback model after it, declined to review the game."""


def review_game(
    client: Any,
    card: Card,
    game: Game,
    game_id: str,
    asks: list[dict[str, Any]],
    opponent_cards: dict[int, Card] | None = None,
) -> dict[str, Any]:
    """Review a game; returns the review as it is kept with the game. ``client`` is an
    ``anthropic.Anthropic``; ``asks`` are the strategist picks kept with the game
    (``mahjong_api.notes``)."""
    langfuse = langfuse_client()
    moves = len(game.events)
    outcome, result, dropped, trace_id = "error", None, None, None
    try:
        with (
            trace(
                "review-game",
                game_id=game_id,
                feature="review",
                version=PROMPT_VERSION,
                prompt=prompt_link(PROMPT_NAME, PROMPT_VERSION),
                moves=moves,
                asks=len(asks),
            ),
            langfuse.start_as_current_observation(
                name="review-game", as_type="evaluator", input={"moves": moves, "asks": asks}
            ) as root,
            marking_errors(root),
        ):
            trace_id = langfuse.get_current_trace_id()
            seen = _replay(langfuse, card, game, asks, opponent_cards or {})
            try:
                answer = _ask_claude(client, langfuse, seen)
            except Refused:
                outcome = "refused"
                raise
            result, dropped = _checked(answer, asks, moves)
            outcome = "reviewed"
            result |= {
                "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "moves": moves,
                "trace_id": trace_id,
                "helpful": None,
            }
            root.update(output=result, metadata={"dropped_refs": dropped})
    finally:
        _score(langfuse, game_id, trace_id, outcome, result, dropped, asks)
        flush()
    return result


def _replay(
    langfuse: Any, card: Card, game: Game, asks: list[dict[str, Any]], opponent_cards: dict
) -> dict[str, Any]:
    """The engine's facts about the game, with what the strategist said added to each pick."""
    picks = [Pick(a["ask"], a["move"], a["hand_id"], a.get("followed")) for a in asks]
    with langfuse.start_as_current_observation(
        name="hindsight", as_type="span", input={"moves": len(game.events), "picks": len(picks)}
    ) as span:
        seen = facts(card, game, picks, opponent_cards)
        said = {a["ask"]: a.get("reason") for a in asks}
        for entry in seen["strategist_picks"]:
            entry["strategist_said"] = said.get(entry["ask"])
        span.update(
            output={
                key: len(seen[key])
                for key in (
                    "charleston_passes",
                    "your_discards",
                    "call_chances",
                    "missed_mahjong",
                    "strategist_picks",
                )
            }
        )
    return seen


def _ask_claude(client: Any, langfuse: Any, seen: dict[str, Any]) -> dict[str, Any]:
    question = "Here is the game, as the engine replayed it:\n" + json.dumps(seen, indent=1)
    with (
        langfuse.start_as_current_observation(
            name="review",
            as_type="generation",
            model=MODEL,
            model_parameters={"effort": EFFORT, "max_tokens": MAX_TOKENS},
            input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        ) as generation,
        marking_errors(generation),
    ):
        # Streamed: with thinking, a long game can take a while, and a stream never times out
        # waiting for the first byte.
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": question}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            # If a safety classifier ever declines, retry on Anthropic's recommended model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
        text = next((b.text for b in response.content if b.type == "text"), "")
        record_claude(generation, response, MODEL, text)
        if response.stop_reason == "refusal":
            raise Refused("Claude declined to review this game")
        if response.stop_reason == "max_tokens":
            raise ReviewError("The review was cut off; try again")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ReviewError("The review came back unreadable; try again") from e


def _checked(
    answer: dict[str, Any], asks: list[dict[str, Any]], moves: int
) -> tuple[dict[str, Any], int]:
    """The review as shown and kept, with anything that cites a move or an ask that doesn't
    exist left out. Returns it and how many points were dropped."""
    score = answer.get("play_score")
    if not isinstance(score, int) or not 1 <= score <= 10:
        raise ReviewError(f"The review gave a score of {score!r}, not 1 to 10")
    dropped = 0
    moments = []
    for moment in answer.get("moments", []):
        if isinstance(moment.get("move"), int) and 1 <= moment["move"] <= moves:
            moments.append(moment)
        else:
            dropped += 1
    moments.sort(key=lambda moment: moment["move"])
    by_number = {a["ask"]: a for a in asks}
    verdicts: dict[int, dict[str, Any]] = {}
    for verdict in answer.get("strategist", []):
        ask = by_number.get(verdict.get("ask"))
        if ask is None or ask["ask"] in verdicts or verdict.get("verdict") not in VERDICTS:
            dropped += 1
            continue
        verdicts[ask["ask"]] = {
            "ask": ask["ask"],
            "move": ask["move"],
            "hand_id": ask["hand_id"],
            "label": ask.get("label"),
            "followed": ask.get("followed"),
            "verdict": verdict["verdict"],
            "why": verdict.get("why", ""),
        }
    review = {
        "play_score": score,
        "summary": answer.get("summary", ""),
        "went_well": list(answer.get("went_well", []))[:WENT_WELL],
        "moments": moments[:MOMENTS],
        "strategist": [verdicts[n] for n in sorted(verdicts)],
        "strategist_summary": answer.get("strategist_summary", "") if asks else "",
        "work_on": answer.get("work_on", ""),
        "model": MODEL,
    }
    return review, dropped


def _score(
    langfuse: Any,
    game_id: str,
    trace_id: str | None,
    outcome: str,
    result: dict[str, Any] | None,
    dropped: int | None,
    asks: list[dict[str, Any]],
) -> None:
    """One score each per review, the play score on the game and a verdict on each strategist
    pick's own trace. Reviewing again overwrites them. Never raises."""
    if not trace_id:
        return
    try:
        record_score(
            langfuse, "review_outcome", outcome, trace_id=trace_id, score_id=f"{trace_id}-outcome"
        )
        if result is None:
            return
        record_score(
            langfuse,
            "review_dropped_refs",
            dropped or 0,
            trace_id=trace_id,
            score_id=f"{trace_id}-dropped-refs",
        )
        record_score(
            langfuse,
            "review_play_score",
            result["play_score"],
            session_id=game_id,
            score_id=f"{game_id}-review_play_score",
            comment=result["summary"][:500] or None,
            metadata={"review_trace": trace_id},
        )
        traces = {a["ask"]: a.get("trace_id") for a in asks}
        for verdict in result["strategist"]:
            pick_trace = traces.get(verdict["ask"])
            if not pick_trace:
                continue
            record_score(
                langfuse,
                "strategist_hindsight",
                verdict["verdict"],
                trace_id=pick_trace,
                score_id=f"{pick_trace}-strategist_hindsight",
                comment=verdict["why"][:500] or None,
                metadata={"game": game_id, "review_trace": trace_id, "ask": verdict["ask"]},
            )
    except Exception:
        log.warning("Couldn't score a game review", exc_info=True)


def score_helpful(trace_id: str, useful: bool) -> None:
    """The player's "Useful" / "Not useful" tap on a review. Changing your mind overwrites it."""
    record_score(
        langfuse_client(),
        "review_helpful",
        useful,
        trace_id=trace_id,
        score_id=f"{trace_id}-review_helpful",
    )
    flush()
