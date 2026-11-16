# Showcase copy of services/api/src/mahjong_api/selftest.py from the private mahjong-companion repo. Here to be read, not run.

"""Check the Langfuse wiring end to end, without touching what the dashboards show (ADR 7).

One made-up action per feature goes through the app's real code (strategist, photo reading,
table talk, play-by-play), with scripted stand-ins for Claude and Jev so it costs nothing. The
traces are sent to the real Langfuse project under the ``selftest`` environment, which no
dashboard or report reads, then read back and checked: are they in the right session and
environment, tagged, versioned, linked to their prompt, priced once, and scored?

What a run leaves behind: about 10 observations and 18 scores in the ``selftest`` environment.
Langfuse is asked to delete the traces, but it only queues that request, and it allows just 50
score deletions a day, so the scores are left where they are.

Run by ``scripts/langfuse_smoke.py``. Worth running after changing keys, the Langfuse SDK or
anything in ``observability``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from mahjong_api import evalset, history, listen, playbyplay, strategist, vision
from mahjong_api.observability import flush, record_score

ENVIRONMENT = "selftest"
WAIT = 90  # seconds to wait for Langfuse to show what was sent


def _usage(input_tokens: int = 1000, output_tokens: int = 50) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


class _Claude:
    """Replays scripted replies in place of the Anthropic client."""

    def __init__(self, *replies: Any) -> None:
        self._replies = list(replies)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        return self._replies.pop(0)


def _tool_call(name: str, **tool_input: Any) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", id=f"call-{name}", name=name, input=tool_input)
    return SimpleNamespace(
        content=[block], stop_reason="tool_use", usage=_usage(), model=strategist.MODEL
    )


class _Jev:
    def system_one(self, state: dict, questions: dict) -> SimpleNamespace:
        return SimpleNamespace(
            model="jev-1.13.0",
            usage={"input_tokens": 400, "output_tokens": 0},
            answers={
                "game_talk": {"type": "noul", "noul": 0.9},
                "tile": {
                    "type": "choice",
                    "choice": "white dragon",
                    "probabilities": {"white dragon": 0.8},
                },
            },
        )


def send(langfuse: Any, session: str) -> dict[str, str]:
    """One made-up action per feature. Returns the trace ID of each."""
    card = evalset.sample_card()
    position = evalset.POSITIONS["triple-threes"]
    state = position.state()
    closest = evalset.candidates(card, state)[0].hand.id
    pick = strategist.ask(
        _Claude(
            _tool_call("opponent_reads"),
            _tool_call("submit_pick", hand_id=closest, reason="Self-test: go for the closest."),
        ),
        card,
        state,
        session,
    )
    record_score(
        langfuse,
        "recommendation_followed",
        True,
        trace_id=pick.trace_id,
        score_id=f"{pick.trace_id}-recommendation_followed",
    )

    tiles = {"tiles": [{"tile": "3 bam", "uncertain": False}, {"tile": "joker", "uncertain": True}]}
    reply = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(tiles))],
        stop_reason="end_turn",
        usage=_usage(1600, 200),
        model=vision.MODEL,
    )
    reading = vision.read_hand(_Claude(reply), "c2VsZi10ZXN0", "image/jpeg", session)
    vision.score_reading(reading.trace_id, reading.tiles, ["3 bam", "flower"])

    heard = listen.hear(state, "I'm getting rid of the blank one", session, _Jev())
    listen.score(heard["trace_id"], "confirmed")

    view = {
        "events": [
            {"type": "discard", "player": 2, "tile": "3 bam", "source": "auto"},
            {"type": "mahjong", "player": 1},
        ],
        "winner": 1,
        "players": 2,
        "turn": 1,
        "phase": "over",
    }
    playbyplay.record_move(session, 1, {"type": "mahjong", "player": 1}, view)
    flush()
    return {"strategist": pick.trace_id, "read-hand": reading.trace_id, "heard": heard["trace_id"]}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


def read_back(langfuse: Any, session: str, since: datetime) -> tuple[list[Any], list[Any]]:
    """What Langfuse has for the session, waiting until it has all of it (or time runs out)."""
    deadline = time.monotonic() + WAIT
    observations: list[Any] = []
    scores: list[Any] = []
    while time.monotonic() < deadline:
        observations = list(
            history.observations(langfuse, since=since, session_id=session, fields=history.FIELDS)
        )
        scores = list(history.scores(langfuse, since=since, environments=[ENVIRONMENT]))
        if len(observations) >= EXPECTED_OBSERVATIONS and len(scores) >= len(EXPECTED_SCORES):
            break
        time.sleep(5)
    return observations, scores


EXPECTED_OBSERVATIONS = 10
EXPECTED_SCORES = {
    "strategist_outcome",
    "strategist_llm_calls",
    "strategist_tool_calls",
    "strategist_rejected_picks",
    "pick_engine_rank",
    "reason_words",
    "reason_within_limit",
    "recommendation_followed",
    "tiles_read",
    "tiles_uncertain",
    "tile_cv_correct",
    "tile_fixes",
    "discard_parse_correct",
    "heard_outcome",
    "game_won",
    "game_moves",
    "game_fixed_moves",
    "discards_auto_logged",
}


def check(observations: list[Any], scores: list[Any], session: str) -> list[Check]:
    """What should be true of the traces that came back."""
    by_name: dict[str, list[Any]] = {}
    for o in observations:
        by_name.setdefault(o.name, []).append(o)
    out = []

    def expect(name: str, passed: bool, detail: str = "") -> None:
        out.append(Check(name, bool(passed), detail))

    names = sorted(by_name)
    expect(
        "every step arrived",
        {"strategist", "claude", "opponent_reads", "read-hand", "heard", "parse-line", "jev"}
        <= set(names),
        f"{len(observations)} observations: {', '.join(names)}",
    )
    expect(
        "all in the self-test environment and this session",
        observations
        and all(o.environment == ENVIRONMENT and o.session_id == session for o in observations),
    )
    copies = [o.name for o in observations if o.name.startswith("anthropic.")]
    expect("each model call recorded once", not copies, ", ".join(copies))
    tags = {o.name: (getattr(o, "tags", None) or []) for o in observations}
    expect(
        "tagged by feature",
        tags.get("strategist") == ["strategist"]
        and tags.get("read-hand") == ["vision"]
        and tags.get("heard") == ["table-talk"],
        str({k: tags.get(k) for k in ("strategist", "read-hand", "heard")}),
    )
    claude = by_name.get("claude", [None])[0]
    expect(
        "the strategist's calls carry its prompt version",
        claude is not None and claude.version == strategist.PROMPT_VERSION,
        getattr(claude, "version", ""),
    )
    expect(
        "the strategist's calls link to the registered prompt",
        claude is not None and getattr(claude, "prompt_name", None) == strategist.PROMPT_NAME,
        "not linked: run scripts/langfuse_setup.py to register the prompts",
    )
    expect(
        "Claude's tokens are priced",
        claude is not None and (claude.total_cost or 0) > 0,
        f"cost {getattr(claude, 'total_cost', None)}",
    )
    jev_call = by_name.get("jev", [None])[0]
    expect(
        "Jev's call carries its tokens and cost",
        jev_call is not None
        and (jev_call.usage_details or {}).get("input") == 400
        and (jev_call.total_cost or 0) > 0,
    )
    photo = by_name.get("read-hand", [None])[0]
    expect(
        "the photo itself was not sent",
        photo is not None and "c2VsZi10ZXN0" not in (photo.input or ""),
    )
    found = {score.name for score in scores}
    missing = sorted(EXPECTED_SCORES - found)
    expect("every score arrived", not missing, f"missing: {', '.join(missing)}")
    game = [s for s in scores if s.name == "game_won"]
    expect(
        "whole-game scores sit on the session",
        game and history.subject(game[0]) == ("session", session),
    )
    return out


def clean_up(langfuse: Any, observations: list[Any]) -> str:
    """Ask Langfuse to delete the self-test's traces, and say how that went. Never raises: a
    failed clean-up must not hide the checks. Langfuse queues the request and removes the traces
    later; scores are not deleted one by one, because Langfuse allows only 50 of those a day."""
    traces = sorted({o.trace_id for o in observations})
    if not traces:
        return "Nothing to clean up."
    try:
        langfuse.api.trace.delete_multiple(trace_ids=traces)
    except Exception as e:
        return f"Couldn't ask Langfuse to delete the self-test's {len(traces)} traces: {e}"
    return (
        f"Asked Langfuse to delete the self-test's {len(traces)} traces. Until it does, they "
        f"sit in the {ENVIRONMENT!r} environment, which no dashboard reads."
    )


def run(langfuse: Any) -> tuple[list[Check], str]:
    """Send, read back, check and clean up. Returns the checks and a note on the clean-up.
    The caller must have set the ``selftest`` environment before the client was first used."""
    started = datetime.now(UTC) - timedelta(minutes=1)
    session = f"selftest-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    send(langfuse, session)
    observations, scores = read_back(langfuse, session, started)
    try:
        checks = check(observations, scores, session)
    finally:
        note = clean_up(langfuse, observations)
    return checks, note
