# Showcase copy of services/api/src/mahjong_api/observability.py from the private mahjong-companion repo. Here to be read, not run.

"""Shared Langfuse plumbing: one client, one way to label a trace, one list of scores (ADR 7).

Every AI step in the app records the same things the same way, so the Langfuse dashboards and the
drift report can compare like with like:

- **Environment**: ``production`` (the Container App), ``development`` (a laptop), ``eval``
  (scripts that measure a model on fixed inputs) and ``selftest`` (the wiring check). Real games
  are never mixed with test runs.
- **Release**: the git commit of the image (``LANGFUSE_RELEASE``, set when the image is built).
- **Feature tag and session**: each trace is tagged with its feature and grouped by game.
- **Prompt version**: a fingerprint of the prompt text, so a quality change can be tied to a
  prompt change. When the prompt is registered in Langfuse (``scripts/langfuse_setup.py``) the
  generation links to it and Langfuse shows cost, latency and scores per prompt version.
- **Scores**: every score the app writes is declared in ``SCORES`` with its type and meaning.

The Anthropic SDK emits its own OpenTelemetry span for each API call. Langfuse would store that
as a second generation and count the call's cost twice, so those spans are not exported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.span_filter import is_default_export_span

log = logging.getLogger(__name__)

SERVICE = "mahjong-api"
ANTHROPIC_SDK_SCOPE = "com.anthropic.sdk.python"

# The signed-in player for the current request, as a short hash (set by main.PlayerMiddleware).
_player: ContextVar[str | None] = ContextVar("mahjong_player", default=None)


def environment() -> str:
    """Where this process runs. ``LANGFUSE_TRACING_ENVIRONMENT`` wins: the deploy builds the
    image with it set to production, and scripts set their own. Without it, a Container App is
    still production (Azure sets ``CONTAINER_APP_NAME`` there) and anything else development."""
    if name := os.environ.get("LANGFUSE_TRACING_ENVIRONMENT"):
        return name
    return "production" if os.environ.get("CONTAINER_APP_NAME") else "development"


def use_environment(name: str) -> None:
    """For scripts: trace this run under ``name`` unless the caller already chose one. Call it
    before the first ``langfuse_client()``."""
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", name)


def _should_export(span: Any) -> bool:
    """Everything Langfuse exports by default, minus the Anthropic SDK's own span per API call
    (the app records each call itself, with its input and output)."""
    scope = getattr(span, "instrumentation_scope", None)
    name = getattr(scope, "name", "") or ""
    if name == ANTHROPIC_SDK_SCOPE or name.startswith(ANTHROPIC_SDK_SCOPE + "."):
        return False
    return is_default_export_span(span)


@cache
def langfuse_client() -> Langfuse:
    """The Langfuse client for this app's project.

    ``LANGFUSE_PUBLIC_KEY_MJ`` / ``LANGFUSE_SECRET_KEY_MJ`` win when set, so a machine whose
    plain ``LANGFUSE_*`` keys belong to another project (Claude Code tracing) still traces here.
    In Azure the plain names come from Key Vault. Without keys the client is switched off and
    every call below does nothing.
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE)
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY_MJ")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY_MJ")
    if not public_key:
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key:
        return get_client()
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
        environment=environment(),
        should_export_span=_should_export,
    )


def flush() -> None:
    """Send what is queued to Langfuse now. Never raises: tracing must not break a move."""
    try:
        langfuse_client().flush()
    except Exception:
        log.warning("Langfuse flush failed", exc_info=True)


def set_player(principal: str | None) -> Any:
    """Remember who is playing for this request, as a hash: the sign-in ID never leaves the app."""
    hashed = f"player-{hashlib.sha256(principal.encode()).hexdigest()[:10]}" if principal else None
    return _player.set(hashed)


def reset_player(token: Any) -> None:
    _player.reset(token)


def prompt_version(*parts: Any) -> str:
    """A short fingerprint of a prompt (text, tools, schema): it changes whenever they do."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def prompt_label(version: str) -> str:
    """The Langfuse prompt label that points at the version with this fingerprint."""
    return f"sha-{version}"


@contextmanager
def sdk_quiet() -> Iterator[None]:
    """Silence the Langfuse SDK's own log for a lookup that is expected to miss sometimes: its
    "not found" warning reads like a failure."""
    sdk_log = logging.getLogger("langfuse")
    level = sdk_log.level
    sdk_log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        sdk_log.setLevel(level)


@cache
def prompt_link(name: str, version: str) -> dict[str, Any] | None:
    """The Langfuse prompt registered for this fingerprint, so generations can link to it.

    The prompt text lives in code; ``scripts/langfuse_setup.py`` mirrors it into Langfuse under a
    ``sha-<fingerprint>`` label. One lookup per process and prompt; an unregistered prompt or an
    unreachable Langfuse just means no link (the fingerprint is still on the trace).
    """
    try:
        with sdk_quiet():
            found = langfuse_client().get_prompt(
                name,
                label=prompt_label(version),
                cache_ttl_seconds=24 * 3600,
                max_retries=0,
                fetch_timeout_seconds=3,
            )
    except Exception:
        log.info("Prompt %s@%s isn't in Langfuse yet: run scripts/langfuse_setup.py", name, version)
        return None
    return {"name": found.name, "version": found.version}


@contextmanager
def trace(
    name: str,
    *,
    game_id: str,
    feature: str,
    version: str | None = None,
    prompt: dict[str, Any] | None = None,
    **metadata: Any,
) -> Iterator[None]:
    """Label everything recorded inside as one trace: its name, the game it belongs to (the
    Langfuse session), its feature tag, the prompt version, the player and any extra facts."""
    facts = {key: str(value) for key, value in metadata.items() if value is not None}
    if version:
        facts["prompt_version"] = version
    with propagate_attributes(
        session_id=game_id,
        trace_name=name,
        tags=[feature],
        user_id=_player.get(),
        version=version,
        prompt=prompt,
        metadata=facts or None,
    ):
        yield


@contextmanager
def marking_errors(observation: Any) -> Iterator[None]:
    """Mark the observation ERROR, with the reason, when the code inside raises; the error then
    carries on to the caller. Without this a failed call looks like any other in Langfuse."""
    try:
        yield
    except Exception as e:
        observation.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:300])
        raise


def claude_usage(usage: Any) -> dict[str, int]:
    """Token counts in the names Langfuse prices: fresh input, output, cache reads and writes."""
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None) or 0,
    }


def same_model(requested: str, served: str | None) -> bool:
    """True when the model that answered is the one asked for. The API may answer with a dated
    snapshot name ("claude-sonnet-5-5-20260930") for an alias ("claude-sonnet-5-5")."""
    if not served:
        return True
    return served.startswith(requested) or requested.startswith(served)


def record_claude(generation: Any, response: Any, requested: str, output: Any) -> None:
    """Finish a Claude generation: its output, tokens, why it stopped and which model answered.

    A model other than the one requested means the server-side refusal fallback stepped in: the
    generation is then priced as that model and marked WARNING so it shows up in Langfuse.
    """
    usage = response.usage
    served = getattr(response, "model", None)
    fell_back = not same_model(requested, served)
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    details = getattr(usage, "output_tokens_details", None)
    facts = {
        "stop_reason": response.stop_reason,
        "requested_model": requested,
        "served_model": served or requested,
        "fallback": fell_back,
        "cache_hit": cache_read > 0,
        "thinking_tokens": getattr(details, "thinking_tokens", None),
        "service_tier": getattr(usage, "service_tier", None),
        "response_id": getattr(response, "id", None),
        "request_id": getattr(response, "_request_id", None),
    }
    update: dict[str, Any] = {
        "output": output,
        "metadata": {key: value for key, value in facts.items() if value is not None},
    }
    if usage is not None:
        update["usage_details"] = claude_usage(usage)
    if fell_back:
        update |= {
            "model": served,
            "level": "WARNING",
            "status_message": f"Answered by {served}, not {requested}",
        }
    generation.update(**update)


ScoreType = Literal["NUMERIC", "BOOLEAN", "CATEGORICAL"]


@dataclass(frozen=True)
class ScoreDef:
    """One score the app writes: its Langfuse type, what it means and who produces it
    (``player``: a tap; ``code``: a deterministic check; ``review``: Claude's game review)."""

    name: str
    data_type: ScoreType
    source: Literal["player", "code", "review"]
    description: str
    min_value: float | None = None
    max_value: float | None = None
    categories: tuple[str, ...] = ()


_SCORES = [
    # Strategist
    ScoreDef(
        "recommendation_followed",
        "BOOLEAN",
        "player",
        "The player tapped \"I'll go for it\" on the strategist's pick (Phase 4 exit check).",
    ),
    ScoreDef(
        "strategist_outcome",
        "CATEGORICAL",
        "code",
        "How a strategist ask ended.",
        categories=("picked", "refused", "no_pick", "error"),
    ),
    ScoreDef(
        "pick_engine_rank",
        "NUMERIC",
        "code",
        "Where the picked hand stood in the engine's closest-first list (1 = the closest).",
        min_value=1,
        max_value=5,
    ),
    ScoreDef("reason_words", "NUMERIC", "code", "Words in the strategist's reason.", min_value=0),
    ScoreDef(
        "reason_within_limit",
        "BOOLEAN",
        "code",
        "The reason kept to the 25 words the prompt asks for.",
    ),
    ScoreDef(
        "strategist_llm_calls",
        "NUMERIC",
        "code",
        "Claude calls one ask took (1 to 4).",
        min_value=1,
        max_value=4,
    ),
    ScoreDef(
        "strategist_tool_calls",
        "NUMERIC",
        "code",
        "Engine tool calls (hand_details, opponent_reads) one ask made.",
        min_value=0,
    ),
    ScoreDef(
        "strategist_rejected_picks",
        "NUMERIC",
        "code",
        "Times Claude named a hand outside the engine's candidates before settling (the "
        "guard against invented hands at work).",
        min_value=0,
    ),
    # The game review (Claude, once a game is over), on the strategist's picks and the game
    ScoreDef(
        "strategist_hindsight",
        "CATEGORICAL",
        "review",
        "The game review's verdict on a strategist pick, knowing how the game went.",
        categories=("good", "reasonable", "poor"),
    ),
    ScoreDef(
        "review_play_score",
        "NUMERIC",
        "review",
        "The game review's score for your play, 1 to 10, on the whole game.",
        min_value=1,
        max_value=10,
    ),
    ScoreDef(
        "review_outcome",
        "CATEGORICAL",
        "code",
        "How a game review ended.",
        categories=("reviewed", "refused", "error"),
    ),
    ScoreDef(
        "review_dropped_refs",
        "NUMERIC",
        "code",
        "Points in a game review that cited a move or a strategist ask that doesn't exist, "
        "dropped before showing it (the guard against invented moves at work).",
        min_value=0,
    ),
    ScoreDef(
        "review_helpful",
        "BOOLEAN",
        "player",
        'The player tapped "Useful" (or "Not useful") on a game review.',
    ),
    ScoreDef(
        "picked_closest",
        "BOOLEAN",
        "code",
        "Regression set: the strategist picked the hand the engine ranks closest.",
    ),
    # Photo reading
    ScoreDef(
        "tile_cv_correct",
        "NUMERIC",
        "player",
        "Share of the final tiles a photo reading got right.",
        min_value=0,
        max_value=1,
    ),
    ScoreDef(
        "tile_fixes",
        "NUMERIC",
        "player",
        "Tiles the player changed, added or removed after a photo reading (Phase 5 exit "
        "check: at most one on average).",
        min_value=0,
    ),
    ScoreDef("tiles_read", "NUMERIC", "code", "Tiles a photo reading returned.", min_value=0),
    ScoreDef(
        "tiles_uncertain",
        "NUMERIC",
        "code",
        "Tiles a photo reading flagged for the player to check.",
        min_value=0,
    ),
    # Table talk
    ScoreDef(
        "discard_parse_correct",
        "NUMERIC",
        "player",
        "A heard line was read right: confirmed or auto-logged and left alone (1) or fixed or "
        "ignored (0).",
        min_value=0,
        max_value=1,
    ),
    ScoreDef(
        "heard_outcome",
        "CATEGORICAL",
        "player",
        "What the player did with a heard line's proposal.",
        categories=("confirmed", "auto", "fixed", "ignored"),
    ),
    # Card drafting
    ScoreDef(
        "card_draft_accuracy",
        "NUMERIC",
        "code",
        "Share of a known card's hands a draft got exactly right, by meaning.",
        min_value=0,
        max_value=1,
    ),
    ScoreDef(
        "card_draft_rounds",
        "NUMERIC",
        "code",
        "Rounds of validator feedback a card draft took (1 to 4).",
        min_value=1,
        max_value=4,
    ),
    ScoreDef("card_draft_clean", "BOOLEAN", "code", "The draft passed the engine's validator."),
    # Whole games (scores on the session)
    ScoreDef("game_won", "BOOLEAN", "code", "Player 1 won the game."),
    ScoreDef("game_moves", "NUMERIC", "code", "Moves in a finished game.", min_value=0),
    ScoreDef(
        "discards_auto_logged",
        "NUMERIC",
        "code",
        "Share of a game's discards logged from a heard line without a tap (Phase 6 exit "
        "check: most discards logged with no fix).",
        min_value=0,
        max_value=1,
    ),
    ScoreDef(
        "game_fixed_moves",
        "NUMERIC",
        "code",
        "Moves in a finished game that the player had to change afterwards.",
        min_value=0,
    ),
]
SCORES: dict[str, ScoreDef] = {score.name: score for score in _SCORES}


def record_score(
    langfuse: Any,
    name: str,
    value: float | bool | str,
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    observation_id: str | None = None,
    comment: str | None = None,
    score_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    config_id: str | None = None,
    environment: str | None = None,
) -> None:
    """Write one declared score on a trace, an observation or a whole game (the session).

    The type comes from ``SCORES``, so a score can't be written under two types. Nothing is
    written when there is nothing to attach it to (tracing switched off). ``environment`` is
    for scores written from another process than the trace's (a script): it files the score
    with the trace it belongs to instead of with the process that wrote it.
    """
    definition = SCORES[name]
    if not (trace_id or session_id):
        return
    if definition.data_type == "CATEGORICAL":
        if value not in definition.categories:
            raise ValueError(f"{name} must be one of {definition.categories}, not {value!r}")
    else:
        value = float(value)
    fields = {
        "name": name,
        "value": value,
        "data_type": definition.data_type,
        "trace_id": trace_id,
        "session_id": session_id,
        "observation_id": observation_id,
        "comment": comment,
        "score_id": score_id,
        "metadata": metadata,
        "config_id": config_id,
        "environment": environment,
    }
    langfuse.create_score(**{key: field for key, field in fields.items() if field is not None})
