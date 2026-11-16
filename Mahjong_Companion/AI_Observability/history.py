# Showcase copy of services/api/src/mahjong_api/history.py from the private mahjong-companion repo. Here to be read, not run.

"""Read back what the app sent to Langfuse, for the judge and the drift report.

Langfuse's v2 observations and v3 scores endpoints are the ones that still exist for projects
created after September 2026; both page with a cursor. Inputs and outputs come back as JSON
text, so ``loads`` turns them into data again.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

# Everything the judge and the drift report read from an observation.
FIELDS = "core,basic,io,metadata,model,usage,metrics,trace_context,prompt"
PAGE = 200
SCORE_PAGE = 100  # the most the scores endpoint allows
# The SDK gives a read 5 seconds; a page of observations with their inputs can take longer.
PATIENCE = {"timeout_in_seconds": 60}
TRIES = 3


def _page(fetch: Callable[[], Any]) -> Any:
    """One page, tried again after a pause if Langfuse is slow or briefly unavailable."""
    for attempt in range(TRIES):
        try:
            return fetch()
        except Exception:
            if attempt == TRIES - 1:
                raise
            time.sleep(2 * (attempt + 1))


def observations(
    langfuse: Any,
    *,
    since: datetime,
    until: datetime | None = None,
    fields: str = FIELDS,
    **filters: Any,
) -> Iterator[Any]:
    """Every observation that started in the window and matches the filters (``name``,
    ``type``, ``environment``, ``trace_id``, ``session_id``, ...), newest first."""
    cursor = None
    while True:
        page = _page(
            lambda cursor=cursor: langfuse.api.observations.get_many(
                fields=fields,
                limit=PAGE,
                cursor=cursor,
                from_start_time=since,
                to_start_time=until,
                request_options=PATIENCE,
                **{key: value for key, value in filters.items() if value is not None},
            )
        )
        yield from page.data
        cursor = getattr(page.meta, "cursor", None)
        if not cursor or not page.data:
            return


def scores(
    langfuse: Any,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    names: list[str] | None = None,
    environments: list[str] | None = None,
) -> Iterator[Any]:
    """Every score in the window, with its comment, metadata and what it is attached to."""
    cursor = None
    while True:
        page = _page(
            lambda cursor=cursor: langfuse.api.scores.get_many(
                fields="details,subject",
                limit=SCORE_PAGE,
                cursor=cursor,
                name=",".join(names) if names else None,
                environment=",".join(environments) if environments else None,
                from_timestamp=since,
                to_timestamp=until,
                request_options=PATIENCE,
            )
        )
        yield from page.data
        cursor = getattr(page.meta, "cursor", None)
        if not cursor or not page.data:
            return


def loads(text: Any) -> Any:
    """An observation's input or output as data. Text that isn't JSON comes back unchanged."""
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def subject(score: Any) -> tuple[str | None, str | None]:
    """What a score is attached to: ("trace" | "observation" | "session" | "experiment", id).
    For a score on an observation the id is its trace's."""
    target = getattr(score, "subject", None)
    if target is None:
        return None, None
    get = target.get if isinstance(target, dict) else lambda key: getattr(target, key, None)
    kind = get("kind")
    return kind, (get("trace_id") or get("traceId") or get("id")) if kind else None
