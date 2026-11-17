# Showcase copy of services/api/src/mahjong_api/langfuse_setup.py from the private mahjong-companion repo. Here to be read, not run.

"""What the Langfuse project should contain besides traces, kept in code (ADR 7).

``scripts/langfuse_setup.py`` applies it; running it again changes only what differs.

- **Score configs**: every score in ``observability.SCORES`` with its type, range and meaning,
  so Langfuse explains each score where it shows it and offers it for annotating by hand.
- **Prompts**: each prompt's text, mirrored from code under a ``sha-<fingerprint>`` label.
  The code stays the source of truth; Langfuse can then show cost, latency and scores per
  prompt version, and what any fingerprint on an old trace actually said.
- **Datasets**: the fixed test sets (``experiments``).
- **Dashboards**: three boards next to the game-evening overview: is the AI still good, what
  does it cost, and what do the fixed test sets say. A tile someone added to one of these
  boards by hand is kept, below the tiles this module places; so are the local judge's tiles
  from before it was retired (ADR 8).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import groupby
from typing import Any

from mahjong_api import card_draft, experiments, review, strategist, vision
from mahjong_api.observability import SCORES, ScoreDef, prompt_label, sdk_quiet

REAL_GAMES = ("production", "default")  # "default": recorded before environments were set
TEST_RUNS = ("eval", "sdk-experiment")
PAUSE = 1.2  # seconds between writes: the dashboards API allows only a few a second


# --- Score configs -----------------------------------------------------------------------


def score_config(score: ScoreDef) -> dict[str, Any]:
    """A score's definition in the shape Langfuse's score-config API takes."""
    config: dict[str, Any] = {
        "name": score.name,
        "data_type": score.data_type,
        "description": f"{score.description} (from: {score.source})",
    }
    if score.data_type == "NUMERIC":
        if score.min_value is not None:
            config["min_value"] = score.min_value
        if score.max_value is not None:
            config["max_value"] = score.max_value
    if score.data_type == "CATEGORICAL":
        config["categories"] = [
            {"label": label, "value": float(index)} for index, label in enumerate(score.categories)
        ]
    return config


def sync_score_configs(langfuse: Any) -> dict[str, list[str]]:
    """Create the configs that are missing. A config can't change type once made, so one that
    exists under another type is reported, not touched."""
    existing: dict[str, Any] = {}
    page = 1
    while True:
        found = langfuse.api.score_configs.get(page=page, limit=100)
        existing.update({config.name: config for config in found.data})
        if page >= found.meta.total_pages:
            break
        page += 1
    done: dict[str, list[str]] = {"created": [], "unchanged": [], "mismatched": []}
    for score in SCORES.values():
        current = existing.get(score.name)
        if current is None:
            langfuse.api.score_configs.create(**score_config(score))
            done["created"].append(score.name)
            time.sleep(PAUSE / 4)
        elif str(getattr(current.data_type, "value", current.data_type)) != score.data_type:
            done["mismatched"].append(score.name)
        else:
            done["unchanged"].append(score.name)
    return done


# --- Prompts -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str
    version: str  # the fingerprint traces carry
    config: dict[str, Any]
    tags: tuple[str, ...]


def prompts() -> list[PromptSpec]:
    """Every prompt the app uses, as the code has it now."""
    return [
        PromptSpec(
            strategist.PROMPT_NAME,
            strategist.SYSTEM,
            strategist.PROMPT_VERSION,
            {"model": strategist.MODEL, "effort": strategist.EFFORT, "tools": strategist.TOOLS},
            ("strategist",),
        ),
        PromptSpec(
            vision.PROMPT_NAME,
            vision.PROMPT,
            vision.PROMPT_VERSION,
            {"model": vision.MODEL, "effort": vision.EFFORT, "answer_schema": vision.SCHEMA},
            ("vision",),
        ),
        PromptSpec(
            card_draft.PROMPT_NAME,
            card_draft.SYSTEM,
            card_draft.PROMPT_VERSION,
            {"model": card_draft.MODEL, "effort": card_draft.EFFORT},
            ("card-draft",),
        ),
        PromptSpec(
            review.PROMPT_NAME,
            review.SYSTEM,
            review.PROMPT_VERSION,
            {"model": review.MODEL, "effort": review.EFFORT, "answer_schema": review.SCHEMA},
            ("review",),
        ),
    ]


def sync_prompts(langfuse: Any) -> dict[str, list[str]]:
    """Register each prompt whose current fingerprint Langfuse doesn't have yet. Nothing is
    ever edited: a changed prompt becomes a new version with its own label."""
    done: dict[str, list[str]] = {"created": [], "unchanged": []}
    for spec in prompts():
        label = prompt_label(spec.version)
        try:
            with sdk_quiet():
                langfuse.get_prompt(spec.name, label=label, cache_ttl_seconds=0, max_retries=0)
            done["unchanged"].append(f"{spec.name}@{spec.version}")
            continue
        except Exception:
            pass  # not registered yet
        langfuse.create_prompt(
            name=spec.name,
            prompt=spec.text,
            labels=[label],
            tags=list(spec.tags),
            type="text",
            config=spec.config,
            commit_message=f"{spec.version}: mirrored from the code",
        )
        done["created"].append(f"{spec.name}@{spec.version}")
        time.sleep(PAUSE / 4)
    return done


def sync_datasets(langfuse: Any) -> dict[str, int]:
    return {name: experiments.sync_dataset(langfuse, name) for name in experiments.DATASETS}


# --- Dashboards --------------------------------------------------------------------------


def one_of(column: str, *values: str) -> dict[str, Any]:
    return {"column": column, "operator": "any of", "type": "stringOptions", "value": list(values)}


def scores(*names: str) -> tuple[dict[str, Any], ...]:
    return (one_of("name", *names),)


GENERATIONS = one_of("type", "GENERATION")
# The Anthropic SDK's own copy of each call, exported until the app filtered it out: counting
# it would double every cost and token total for those weeks.
NOT_THE_SDK_COPY = {
    "column": "name",
    "operator": "does not contain",
    "type": "string",
    "value": "anthropic.messages",
}
# scripts/vision_eval.py runs from before evals had an environment of their own sit in "default"
# with the real games. Only observations can be told apart by session; a score on a trace has
# no session to filter on, so tile_cv_correct from those runs stays in the real-game averages.
NOT_AN_OLD_EVAL = {
    "column": "sessionId",
    "operator": "does not contain",
    "type": "string",
    "value": "vision-eval",
}
CALLS = (GENERATIONS, NOT_THE_SDK_COPY, NOT_AN_OLD_EVAL)


@dataclass(frozen=True)
class Widget:
    name: str
    description: str
    view: str  # observations, scores-numeric, scores-boolean or scores-categorical
    chart: str
    metrics: tuple[tuple[str, str], ...]  # (measure, aggregation)
    dimensions: tuple[str, ...] = ()
    filters: tuple[dict[str, Any], ...] = ()
    row_limit: int | None = None

    def request(self) -> dict[str, Any]:
        """The widget in the shape the dashboard-widgets API takes."""
        body: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "view": self.view,
            "dimensions": [{"field": name} for name in self.dimensions],
            "metrics": [{"measure": measure, "agg": agg} for measure, agg in self.metrics],
            "filters": [dict(f) for f in self.filters],
            "chartType": self.chart,
        }
        if self.row_limit:
            body["chartConfig"] = {"type": self.chart, "row_limit": self.row_limit}
        return body

    def query(self, board_filters: list[dict[str, Any]], days: int = 30) -> dict[str, Any]:
        """The same question put to the metrics API, to check a widget before creating it."""
        until = datetime.now(UTC)
        query: dict[str, Any] = {
            "view": self.view,
            "dimensions": [{"field": name} for name in self.dimensions],
            "metrics": [{"measure": measure, "aggregation": agg} for measure, agg in self.metrics],
            "filters": [dict(f) for f in self.filters] + board_filters,
            "fromTimestamp": (until - timedelta(days=days)).isoformat(),
            "toTimestamp": until.isoformat(),
        }
        if self.chart.endswith("TIME_SERIES"):
            query["timeDimension"] = {"granularity": "day"}
        return query


@dataclass(frozen=True)
class Board:
    name: str
    description: str
    environments: tuple[str, ...]
    rows: tuple[tuple[int, tuple[Widget, ...]], ...]  # (row height, widgets sharing the row)

    def widgets(self) -> list[Widget]:
        return [widget for _, row in self.rows for widget in row]

    def placements(self, ids: dict[str, str]) -> list[dict[str, Any]]:
        """Each row's widgets side by side, equal widths, on Langfuse's 12-column grid."""
        tiles, y = [], 0
        for height, row in self.rows:
            width = 12 // len(row)
            for column, widget in enumerate(row):
                tiles.append(
                    {
                        "type": "widget",
                        # The tile's own ID: the same on every run, so a rerun changes nothing.
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.name}/{widget.name}")),
                        "widgetId": ids[widget.name],
                        "x": column * width,
                        "y": y,
                        "width": width,
                        "height": height,
                    }
                )
            y += height
        return tiles

    def filters(self) -> list[dict[str, Any]]:
        """The filter on the whole board: which environments it shows."""
        return [one_of("environment", *self.environments)]


def _number(name: str, description: str, view: str, score: str) -> Widget:
    return Widget(name, description, view, "NUMBER", (("value", "avg"),), filters=scores(score))


QUALITY = Board(
    "Mahjong: AI quality",
    "Is the AI still doing a good job? The game review's verdicts on strategist picks, the "
    "checks that need no judgment and your own taps, for real games only.",
    REAL_GAMES,
    (
        (
            3,
            (
                _number(
                    "Reviews you found useful",
                    'Share of game reviews you tapped "Useful" on (review_helpful).',
                    "scores-boolean",
                    "review_helpful",
                ),
                _number(
                    "Picks you followed",
                    'Share of strategist picks you tapped "I\'ll go for it" on '
                    "(recommendation_followed). Phase 4 exit check.",
                    "scores-boolean",
                    "recommendation_followed",
                ),
                _number(
                    "Reasons within 25 words",
                    "Share of strategist reasons that kept to the prompt's word limit "
                    "(reason_within_limit).",
                    "scores-boolean",
                    "reason_within_limit",
                ),
                _number(
                    "Fixes per photo",
                    "Tiles you changed after a photo reading, on average (tile_fixes). Phase 5 "
                    "exit check: at most one.",
                    "scores-numeric",
                    "tile_fixes",
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Strategist picks in hindsight",
                    "The game review's verdict on each strategist pick once the game was over: "
                    "good, reasonable or poor (strategist_hindsight). More poor is the first "
                    "sign of drift.",
                    "scores-categorical",
                    "PIE",
                    (("count", "count"),),
                    ("stringValue",),
                    scores("strategist_hindsight"),
                ),
                Widget(
                    "Your taps over time",
                    "Daily average of the scores your own taps produce: picks followed, tiles "
                    "read right, heard lines read right.",
                    "scores-numeric",
                    "LINE_TIME_SERIES",
                    (("value", "avg"),),
                    ("name",),
                    scores("recommendation_followed", "tile_cv_correct", "discard_parse_correct"),
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "How strategist asks end",
                    "picked, refused, no_pick (call limit reached) or error (strategist_outcome).",
                    "scores-categorical",
                    "PIE",
                    (("count", "count"),),
                    ("stringValue",),
                    scores("strategist_outcome"),
                ),
                Widget(
                    "Engine rank of the pick",
                    "How often the strategist picks the engine's closest hand (1) or a "
                    "farther one (2 to 5) (pick_engine_rank).",
                    "scores-numeric",
                    "VERTICAL_BAR",
                    (("count", "count"),),
                    ("value",),
                    scores("pick_engine_rank"),
                ),
                Widget(
                    "What you did with heard lines",
                    "auto (logged without asking), confirmed, fixed or ignored (heard_outcome). "
                    "Phase 6 exit check: mostly auto and confirmed.",
                    "scores-categorical",
                    "PIE",
                    (("count", "count"),),
                    ("stringValue",),
                    scores("heard_outcome"),
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Your play, game by game",
                    "Daily average of the game review's 1-10 score for your play "
                    "(review_play_score).",
                    "scores-numeric",
                    "LINE_TIME_SERIES",
                    (("value", "avg"),),
                    filters=scores("review_play_score"),
                ),
                Widget(
                    "Strategist effort per ask",
                    "Daily average Claude calls, engine tool calls and rejected (invented) "
                    "picks per ask.",
                    "scores-numeric",
                    "LINE_TIME_SERIES",
                    (("value", "avg"),),
                    ("name",),
                    scores(
                        "strategist_llm_calls",
                        "strategist_tool_calls",
                        "strategist_rejected_picks",
                    ),
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Discards logged without a tap",
                    "Per finished game, the share of discards auto-logged from a heard line "
                    "(discards_auto_logged), with moves fixed afterwards as a count.",
                    "scores-numeric",
                    "LINE_TIME_SERIES",
                    (("value", "avg"),),
                    ("name",),
                    scores("discards_auto_logged", "game_fixed_moves"),
                ),
                Widget(
                    "Photo readings: flagged and fixed",
                    "Daily average of tiles a reading flagged as uncertain against tiles you "
                    "actually fixed: is it unsure about the right tiles?",
                    "scores-numeric",
                    "LINE_TIME_SERIES",
                    (("value", "avg"),),
                    ("name",),
                    scores("tiles_uncertain", "tile_fixes"),
                ),
            ),
        ),
    ),
)

COST = Board(
    "Mahjong: cost and tokens",
    "What the AI costs and how much it reads and writes, for real games only. Each model call "
    "is counted once.",
    REAL_GAMES,
    (
        (
            3,
            (
                Widget(
                    "AI cost in USD",
                    "Claude and Jev, every feature.",
                    "observations",
                    "NUMBER",
                    (("totalCost", "sum"),),
                    filters=CALLS,
                ),
                Widget(
                    "Model calls",
                    "Calls to Claude and Jev.",
                    "observations",
                    "NUMBER",
                    (("count", "count"),),
                    filters=CALLS,
                ),
                Widget(
                    "Tokens in",
                    "Prompt tokens sent, cached ones included.",
                    "observations",
                    "NUMBER",
                    (("inputTokens", "sum"),),
                    filters=CALLS,
                ),
                Widget(
                    "Tokens out",
                    "Tokens the models wrote.",
                    "observations",
                    "NUMBER",
                    (("outputTokens", "sum"),),
                    filters=CALLS,
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Cost per day by call",
                    "claude: the strategist and card drafting; read-hand: photo reading; "
                    "review: game reviews; jev: unclear table talk.",
                    "observations",
                    "BAR_TIME_SERIES",
                    (("totalCost", "sum"),),
                    ("name",),
                    CALLS,
                ),
                Widget(
                    "Cost by feature",
                    "Cost by the kind of trace the call belongs to: strategist, read-hand, "
                    "heard, draft-card, review-game.",
                    "observations",
                    "HORIZONTAL_BAR",
                    (("totalCost", "sum"),),
                    ("traceName",),
                    CALLS,
                    row_limit=10,
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Tokens per call over time",
                    "Daily average of tokens in plus out per call. A step up with no prompt "
                    "change means the model or the games changed.",
                    "observations",
                    "LINE_TIME_SERIES",
                    (("totalTokens", "avg"),),
                    ("name",),
                    CALLS,
                ),
                Widget(
                    "Latency per call, 95th percentile",
                    "In milliseconds, per day: how slow the slow calls are.",
                    "observations",
                    "LINE_TIME_SERIES",
                    (("latency", "p95"),),
                    ("name",),
                    CALLS,
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Prompt versions in use",
                    "Calls, average tokens and average cost for each prompt fingerprint. A new "
                    "row is a prompt change.",
                    "observations",
                    "PIVOT_TABLE",
                    (("count", "count"), ("totalTokens", "avg"), ("totalCost", "avg")),
                    ("name", "version"),
                    CALLS,
                    row_limit=20,
                ),
                Widget(
                    "Cost by model",
                    "Claude Sonnet, Claude Opus and Jev. Another model here means a fallback "
                    "answered.",
                    "observations",
                    "PIE",
                    (("totalCost", "sum"),),
                    ("providedModelName",),
                    CALLS,
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Calls that failed or fell back",
                    "ERROR: the call failed or was refused. WARNING: another model answered, "
                    "or a tool call was rejected.",
                    "observations",
                    "BAR_TIME_SERIES",
                    (("count", "count"),),
                    ("level",),
                    (NOT_THE_SDK_COPY, NOT_AN_OLD_EVAL, one_of("level", "ERROR", "WARNING")),
                ),
                Widget(
                    "Model calls per day",
                    "How much each model call is used.",
                    "observations",
                    "BAR_TIME_SERIES",
                    (("count", "count"),),
                    ("name",),
                    CALLS,
                ),
            ),
        ),
    ),
)

TRUST = Board(
    # The name it was created under: boards are matched by name, so renaming would make another.
    "Mahjong: judge and evals",
    "What do the fixed test sets say? Regression runs and eval scripts only: no real games. "
    "The judge tiles below these are from the local judge retired in ADR 8 and get no new data.",
    TEST_RUNS,
    (
        (
            3,
            (
                _number(
                    "Fixed positions: closest picked",
                    "Regression set: share of the nine fixed positions where the strategist "
                    "picked the engine's closest hand (picked_closest).",
                    "scores-boolean",
                    "picked_closest",
                ),
                _number(
                    "Single-tile photos read right",
                    "scripts/vision_eval.py: average tile_cv_correct on the labelled tile photos.",
                    "scores-numeric",
                    "tile_cv_correct",
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Fixed positions over time",
                    "Each regression run on the same nine positions. The inputs never change, "
                    "so a move here is the model or the prompt.",
                    "scores-numeric",
                    "LINE_TIME_SERIES",
                    (("value", "avg"),),
                    ("name",),
                    scores("picked_closest", "reason_within_limit"),
                ),
                Widget(
                    "Cost of test runs",
                    "What the eval scripts and regression runs spent on Claude.",
                    "observations",
                    "BAR_TIME_SERIES",
                    (("totalCost", "sum"),),
                    ("name",),
                    (GENERATIONS,),
                ),
            ),
        ),
        (
            6,
            (
                Widget(
                    "Test runs per day",
                    "Model calls made by eval scripts and regression runs.",
                    "observations",
                    "BAR_TIME_SERIES",
                    (("count", "count"),),
                    ("name",),
                    (GENERATIONS,),
                ),
            ),
        ),
    ),
)

BOARDS = (QUALITY, COST, TRUST)

# The game-evening overview that already existed. It is corrected, not redrawn.
OVERVIEW = "Mahjong Companion"
OVERVIEW_COST_WIDGETS = ("AI cost (USD)", "AI cost by model")
# Its one chart of "each score" was drawn when every score was a 0-1 accuracy.
OVERVIEW_ACCURACY_CHART = "Accuracy over time"
ACCURACY_SCORES = ("discard_parse_correct", "tile_cv_correct", "recommendation_followed")


def _everything(list_page: Any) -> list[Any]:
    out, page = [], 1
    while True:
        found = list_page(page=page, limit=100)
        out.extend(found.data)
        if page >= found.meta.total_pages:
            return out
        page += 1


def _plain(value: Any) -> Any:
    """API objects as plain data, for comparing with what the code wants."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _same_widget(current: Any, wanted: dict[str, Any]) -> bool:
    have = _plain(current)
    return all(
        have.get(key) == wanted[key]
        for key in ("description", "view", "dimensions", "metrics", "filters", "chartType")
    )


def check_boards(langfuse: Any) -> list[str]:
    """Ask the metrics API each widget's question (read-only). Returns the problems found."""
    import json

    problems = []
    for board in BOARDS:
        for widget in board.widgets():
            try:
                langfuse.api.metrics.metrics(query=json.dumps(widget.query(board.filters())))
            except Exception as e:
                problems.append(f"{board.name} / {widget.name}: {str(e)[:300]}")
            time.sleep(PAUSE / 2)
    return problems


TILE_KEYS = ("type", "id", "widgetId", "x", "y", "width", "height")


def with_kept_tiles(layout: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict]:
    """``layout`` (this module's tiles) followed by the tiles already on the board that show a
    widget this module doesn't place there: in their own order and columns, moved below, one
    row after another with no gaps."""
    ours = {tile["widgetId"] for tile in layout}
    kept = [
        {key: tile.get(key) for key in TILE_KEYS}
        for tile in current
        if tile.get("widgetId") not in ours
    ]
    kept.sort(key=lambda tile: (tile["y"], tile["x"]))
    y = max((tile["y"] + tile["height"] for tile in layout), default=0)
    moved = []
    for _, row in groupby(kept, key=lambda tile: tile["y"]):
        row = list(row)
        moved += [{**tile, "y": y} for tile in row]
        y += max(tile["height"] for tile in row)
    return layout + moved


def sync_boards(langfuse: Any) -> dict[str, list[str]]:
    """Create or update the three boards and their widgets, matched by name. Widgets and
    boards this module doesn't name are left alone, and so are tiles on these boards that it
    doesn't place: they move below its own."""
    api = langfuse.api.unstable
    widgets = {w.name: w for w in _everything(api.dashboard_widgets.list)}
    boards = {b.name: b for b in _everything(api.dashboards.list)}
    done: dict[str, list[str]] = {"created": [], "updated": [], "unchanged": []}
    for board in BOARDS:
        ids = {}
        for widget in board.widgets():
            wanted = widget.request()
            current = widgets.get(widget.name)
            if current is None:
                ids[widget.name] = _write(api.dashboard_widgets.create, wanted).id
                done["created"].append(widget.name)
            elif _same_widget(current, wanted):
                ids[widget.name] = current.id
                done["unchanged"].append(widget.name)
            else:
                _write(api.dashboard_widgets.update, wanted, current.id)
                ids[widget.name] = current.id
                done["updated"].append(widget.name)
        current = boards.get(board.name)
        have = _plain(current) if current is not None else {}
        placed = [
            {key: tile.get(key) for key in TILE_KEYS}
            for tile in have.get("definition", {}).get("widgets", [])
        ]
        layout = {"widgets": with_kept_tiles(board.placements(ids), placed)}
        if current is None:
            _write(
                api.dashboards.create,
                {
                    "name": board.name,
                    "description": board.description,
                    "definition": layout,
                    "filters": board.filters(),
                },
            )
            done["created"].append(board.name)
        else:
            same = (
                placed == layout["widgets"]
                and have.get("description") == board.description
                and have.get("filters") == board.filters()
            )
            if same:
                done["unchanged"].append(board.name)
            else:
                _write(
                    api.dashboards.update,
                    {
                        "description": board.description,
                        "definition": layout,
                        "filters": board.filters(),
                    },
                    current.id,
                )
                done["updated"].append(board.name)
    return done


def _write(call: Any, body: dict[str, Any], target: str | None = None) -> Any:
    """One write to the dashboards API, in its own field names, paced and retried when the
    API says to slow down."""
    fields = {
        "name": body.get("name"),
        "description": body.get("description"),
        "view": body.get("view"),
        "dimensions": body.get("dimensions"),
        "metrics": body.get("metrics"),
        "filters": body.get("filters"),
        "chart_type": body.get("chartType"),
        "chart_config": body.get("chartConfig"),
        "definition": body.get("definition"),
    }
    fields = {key: value for key, value in fields.items() if value is not None}
    for attempt in range(5):
        try:
            result = call(target, **fields) if target else call(**fields)
            time.sleep(PAUSE)
            return result
        except Exception as e:
            if "rate limit" not in str(e).lower() or attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def correct_overview(langfuse: Any) -> list[str]:
    """Corrections to the existing game-evening board, each made once:

    - Its cost tiles summed every observation, including the Anthropic SDK's duplicate of each
      Claude call, so they showed about twice the real cost.
    - Two score tiles filtered on the session, which Langfuse only matches for scores on a whole
      game: every score on a trace was dropped, so the tiles showed no real data.
    - Its chart of every score's daily average is kept to the three 0-1 accuracy scores it was
      drawn for, now that scores on other scales exist.
    - It had no environment filter, so eval runs and self-tests would show up in its game counts
      and accuracy tiles.
    """
    api = langfuse.api.unstable
    board = next((b for b in _everything(api.dashboards.list) if b.name == OVERVIEW), None)
    if board is None:
        return []
    changes = []
    placed = {tile.get("widgetId") for tile in _plain(board)["definition"]["widgets"]}
    for widget in _everything(api.dashboard_widgets.list):
        if widget.id not in placed:
            continue
        before = _plain(widget.filters)
        after, notes = list(before), []
        view = str(getattr(widget.view, "value", widget.view))
        if widget.name in OVERVIEW_COST_WIDGETS and NOT_THE_SDK_COPY not in after:
            after.append(NOT_THE_SDK_COPY)
            notes.append("no longer counts the SDK's copy of each call")
        if view.startswith("scores") and any(f.get("column") == "sessionId" for f in after):
            after = [f for f in after if f.get("column") != "sessionId"]
            notes.append("no longer filters scores by session (that hid every score on a trace)")
        if widget.name == OVERVIEW_ACCURACY_CHART and not any(
            f.get("column") == "name" for f in after
        ):
            after.append(one_of("name", *ACCURACY_SCORES))
            notes.append("shows the three accuracy scores only")
        if after != before:
            _write(api.dashboard_widgets.update, {"filters": after}, widget.id)
            changes.append(f'"{widget.name}" {" and ".join(notes)}')
    if not _plain(board.filters):
        _write(api.dashboards.update, {"filters": [one_of("environment", *REAL_GAMES)]}, board.id)
        changes.append(f'"{OVERVIEW}" now shows real games only ({", ".join(REAL_GAMES)})')
    return changes


@dataclass
class Report:
    score_configs: dict[str, list[str]] = field(default_factory=dict)
    prompts: dict[str, list[str]] = field(default_factory=dict)
    datasets: dict[str, int] = field(default_factory=dict)
    boards: dict[str, list[str]] = field(default_factory=dict)
    overview: list[str] = field(default_factory=list)
