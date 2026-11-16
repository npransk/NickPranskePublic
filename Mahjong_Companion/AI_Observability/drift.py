# Showcase copy of services/api/src/mahjong_api/drift.py from the private mahjong-companion repo. Here to be read, not run.

"""Has the AI changed? Compare a recent window with a baseline, feature by feature (ADR 7).

Models are updated behind a stable name, prompts get edited, and the games themselves vary, so
quality and token use can move without anyone deciding they should. This module reads what the
app recorded in Langfuse and compares two windows of time:

- **Use**: prompt and output tokens, cost and latency per call, and per strategist ask.
- **Quality**: every score, from the player's taps, the checks that need no judgment and the
  local judge.

A change is only called drift when it is both large (15% for use, 10 points for a pass rate) and
unlikely to be chance (a permutation test, corrected for the number of metrics tested at once).
A large change on too little data is reported as "watch", and a metric with fewer than five
values in either window is listed as not having enough data: with a game or two a week, most
metrics start there. Changes of model, prompt version or release between the windows are listed
first, because they are the usual explanation.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from mahjong_api import history

ENVIRONMENTS = ("production", "default")  # "default": traces from before environments were set
# Recorded before evals had an environment of their own; not real games.
LEGACY_EVAL_SESSIONS = {"vision-eval"}
# The Anthropic SDK's duplicate of each call, recorded before the app stopped exporting them.
DUPLICATE_GENERATIONS = {"anthropic.messages.create"}
MIN_VALUES = 5
USE_CHANGE = 0.15  # relative change in tokens, cost or latency that matters
QUALITY_CHANGE = 0.10  # change in a 0-1 score that matters
FALSE_ALARM_RATE = 0.10  # share of flagged metrics allowed to be chance (Benjamini-Hochberg)
ROUNDS = 4000

Worse = Literal["up", "down", None]


@dataclass
class Series:
    """One number tracked over time for one feature."""

    feature: str
    name: str
    worse: Worse  # which direction is bad; None when a change is neither good nor bad
    kind: Literal["use", "quality"]
    unit: str = ""
    scale: Literal["relative", "absolute"] = "relative"  # how a change that matters is measured
    stat: Literal["mean", "median"] = "mean"
    points: list[tuple[datetime, float]] = field(default_factory=list)

    def add(self, at: datetime, value: float | None) -> None:
        if value is not None:
            self.points.append((at, float(value)))


@dataclass(frozen=True)
class Finding:
    feature: str
    name: str
    kind: str
    unit: str
    baseline_n: int
    recent_n: int
    baseline: float | None
    recent: float | None
    change: float | None  # relative, or absolute for 0-1 scores; None without a baseline
    scale: str
    p_value: float | None
    verdict: str  # steady, worse, better, changed, watch, not enough data
    stat: str = "mean"


def _stat(name: str) -> Callable[[list[float]], float]:
    return statistics.median if name == "median" else statistics.fmean


def permutation_p(
    baseline: list[float], recent: list[float], stat: str = "mean", rounds: int = ROUNDS
) -> float:
    """How often shuffling the two windows together gives a gap at least as large as the one
    seen: small means the gap is unlikely to be chance. Makes no assumption about the shape of
    the numbers, which suits token counts and yes/no scores alike. Repeatable (fixed seed)."""
    measure = _stat(stat)
    seen = abs(measure(recent) - measure(baseline))
    if seen == 0:
        return 1.0
    pooled = baseline + recent
    cut = len(recent)
    rng = random.Random(7)
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pooled)
        if abs(measure(pooled[:cut]) - measure(pooled[cut:])) >= seen - 1e-12:
            hits += 1
    return (hits + 1) / (rounds + 1)


def adjusted(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg: with many metrics tested at once, some look significant by chance.
    Returns, for each p-value, the false-alarm rate at which it would just be flagged."""
    count = len(p_values)
    order = sorted(range(count), key=lambda i: p_values[i])
    out = [1.0] * count
    running = 1.0
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * count / rank)
        out[index] = running
    return out


def compare(
    series: list[Series], split: datetime, since: datetime, min_values: int = MIN_VALUES
) -> list[Finding]:
    """Compare each series before and after ``split`` (baseline starts at ``since``)."""
    drafts = []
    for s in series:
        baseline = [v for at, v in s.points if since <= at < split]
        recent = [v for at, v in s.points if at >= split]
        measure = _stat(s.stat)
        before = measure(baseline) if baseline else None
        after = measure(recent) if recent else None
        change = None
        if before is not None and after is not None:
            if s.scale == "absolute":
                change = after - before
            elif before != 0:
                change = (after - before) / abs(before)
            elif after == 0:
                change = 0.0
        enough = len(baseline) >= min_values and len(recent) >= min_values
        p_value = permutation_p(baseline, recent, s.stat) if enough else None
        drafts.append((s, len(baseline), len(recent), before, after, change, p_value))

    tested = [i for i, draft in enumerate(drafts) if draft[6] is not None]
    rates = dict(zip(tested, adjusted([drafts[i][6] for i in tested]), strict=True))
    findings = []
    for i, (s, n_before, n_after, before, after, change, p_value) in enumerate(drafts):
        needed = QUALITY_CHANGE if s.scale == "absolute" else USE_CHANGE
        if p_value is None:
            verdict = "not enough data"
        elif change is None:
            # From exactly zero to something: there is no percentage, but it is a change.
            verdict = "watch" if after else "steady"
        elif abs(change) < needed:
            verdict = "steady"
        elif rates[i] > FALSE_ALARM_RATE:
            verdict = "watch"  # a big gap, but not enough data to rule out chance
        elif s.worse is None:
            verdict = "changed"
        else:
            verdict = "worse" if (change > 0) == (s.worse == "up") else "better"
        findings.append(
            Finding(
                s.feature,
                s.name,
                s.kind,
                s.unit,
                n_before,
                n_after,
                before,
                after,
                change,
                s.scale,
                p_value,
                verdict,
                s.stat,
            )
        )
    return findings


# Score name -> (feature, what it measures, which direction is worse, scale).
SCORES: dict[str, tuple[str, str, Worse, str]] = {
    "recommendation_followed": ("strategist", "picks the player followed", "down", "absolute"),
    "judge_quality": ("strategist", "judge: checks passed", "down", "absolute"),
    "judge_grounded": ("strategist", "judge: reason is grounded", "down", "absolute"),
    "judge_matches_pick": ("strategist", "judge: reason matches the pick", "down", "absolute"),
    "judge_sound_choice": ("strategist", "judge: sound choice", "down", "absolute"),
    "judge_clear": ("strategist", "judge: reason is clear", "down", "absolute"),
    "reason_within_limit": ("strategist", "reasons within the word limit", "down", "absolute"),
    "reason_words": ("strategist", "words per reason", None, "relative"),
    "pick_engine_rank": ("strategist", "engine rank of the pick", None, "relative"),
    "strategist_tool_calls": ("strategist", "tool calls per ask", None, "relative"),
    "strategist_rejected_picks": ("strategist", "invented hands per ask", "up", "relative"),
    "tile_cv_correct": ("photo reading", "tiles read right", "down", "absolute"),
    "tile_fixes": ("photo reading", "fixes per photo", "up", "relative"),
    "tiles_uncertain": ("photo reading", "tiles flagged uncertain", None, "relative"),
    "discard_parse_correct": ("table talk", "heard lines read right", "down", "absolute"),
    "card_draft_accuracy": ("card drafting", "hands drafted right", "down", "absolute"),
    "card_draft_rounds": ("card drafting", "rounds per draft", "up", "relative"),
    "discards_auto_logged": ("games", "discards logged without a tap", "down", "absolute"),
    "game_fixed_moves": ("games", "moves fixed per game", "up", "relative"),
    "game_won": ("games", "games won", None, "absolute"),
}
# Category scores, as the share of one value: score -> [(value, what it measures, worse)].
SHARES: dict[str, list[tuple[str, str, str, Worse]]] = {
    "strategist_outcome": [("picked", "strategist", "asks that ended in a pick", "down")],
    "heard_outcome": [
        ("auto", "table talk", "heard lines logged without asking", "down"),
        ("fixed", "table talk", "heard lines the player had to fix", "up"),
    ],
}


def feature_of(observation: Any) -> str | None:
    """Which feature a generation belongs to, by its name and its trace's."""
    name = observation.name or ""
    if name in DUPLICATE_GENERATIONS:
        return None
    if name == "read-hand":
        return "photo reading"
    if name == "jev":
        return "table talk"
    if name == "judge":
        return "judge"
    if name == "claude" or name.startswith("claude-round"):
        drafting = getattr(observation, "trace_name", None) == "draft-card"
        return "card drafting" if drafting else "strategist"
    return None


@dataclass
class Changes:
    """Models, prompt versions and releases seen for a feature, with when each first appeared."""

    seen: dict[str, dict[str, datetime]] = field(
        default_factory=lambda: defaultdict(dict)
    )  # kind -> value -> first seen

    def note(self, kind: str, value: Any, at: datetime) -> None:
        if value in (None, ""):
            return
        first = self.seen[kind].get(str(value))
        if first is None or at < first:
            self.seen[kind][str(value)] = at

    def new_since(self, split: datetime) -> list[str]:
        """What appeared in the recent window that the baseline didn't have."""
        out = []
        for kind, values in sorted(self.seen.items()):
            before = sorted(v for v, at in values.items() if at < split)
            after = sorted((at, v) for v, at in values.items() if at >= split)
            for at, value in after:
                was = f" (was {', '.join(before)})" if before else ""
                out.append(f"{kind} {value} first seen {at:%Y-%m-%d}{was}")
        return out


def collect(
    langfuse: Any,
    since: datetime,
    until: datetime | None = None,
    environments: tuple[str, ...] = ENVIRONMENTS,
) -> tuple[list[Series], dict[str, Changes]]:
    """Read every generation and score in the window into series, plus what changed."""
    series: dict[tuple[str, str], Series] = {}
    changes: dict[str, Changes] = defaultdict(Changes)

    def track(feature: str, name: str, worse: Worse, kind: str, **more: Any) -> Series:
        key = (feature, name)
        if key not in series:
            series[key] = Series(feature, name, worse, kind, **more)  # type: ignore[arg-type]
        return series[key]

    asks: dict[str, dict[str, Any]] = {}  # strategist trace -> totals
    eval_traces: set[str] = set()  # their scores are skipped too
    for o in history.observations(
        langfuse,
        since=since,
        until=until,
        type="GENERATION",
        environment=list(environments),
        fields="core,basic,model,usage,metrics,metadata,trace_context",
    ):
        if o.session_id in LEGACY_EVAL_SESSIONS:
            eval_traces.add(o.trace_id)
            continue
        feature = feature_of(o)
        if feature is None:
            continue
        at = o.start_time
        usage = o.usage_details or {}
        metadata = o.metadata or {}
        prompt_tokens = sum(
            usage.get(key) or 0
            for key in ("input", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        output_tokens = usage.get("output") or 0
        failed = o.level == "ERROR"
        track(feature, "calls that failed", "up", "use", scale="absolute").add(at, failed)
        if not failed:
            track(feature, "prompt tokens per call", "up", "use", unit="tokens").add(
                at, prompt_tokens
            )
            track(feature, "output tokens per call", "up", "use", unit="tokens").add(
                at, output_tokens
            )
            track(feature, "cost per call", "up", "use", unit="$").add(at, o.total_cost)
            track(feature, "latency per call", "up", "use", unit="s", stat="median").add(
                at, o.latency
            )
        if feature == "strategist":
            ask = asks.setdefault(
                o.trace_id, {"at": at, "calls": 0, "tokens": 0, "cost": 0.0, "cached": 0}
            )
            ask["at"] = min(ask["at"], at)
            ask["calls"] += 1
            ask["tokens"] += prompt_tokens + output_tokens
            ask["cost"] += o.total_cost or 0.0
            ask["cached"] += (usage.get("cache_read_input_tokens") or 0) > 0
        note = changes[feature].note
        note("model", metadata.get("served_model") or o.model, at)
        note("prompt", o.version, at)
        note("release", getattr(o, "release", None), at)

    for ask in asks.values():
        at = ask["at"]
        track("strategist", "Claude calls per ask", "up", "use").add(at, ask["calls"])
        track("strategist", "tokens per ask", "up", "use", unit="tokens").add(at, ask["tokens"])
        track("strategist", "cost per ask", "up", "use", unit="$").add(at, ask["cost"])
        if ask["calls"] > 1:
            # Only a second call can read the cache the first one wrote.
            cached = track(
                "strategist", "later calls served from cache", "down", "use", scale="absolute"
            )
            cached.add(at, ask["cached"] / (ask["calls"] - 1))

    # Heard lines the grammar couldn't place go to Jev: more of them means worse transcripts
    # (or new ways of talking), and more cost.
    for o in history.observations(
        langfuse,
        since=since,
        until=until,
        type="CHAIN",
        name="heard",
        environment=list(environments),
        fields="core,basic,metadata",
    ):
        metadata = o.metadata or {}
        route = metadata.get("route") or metadata.get("decided_by")
        if route:
            sent = track("table talk", "heard lines sent to Jev", "up", "use", scale="absolute")
            sent.add(o.start_time, route != "grammar")

    for score in history.scores(
        langfuse, since=since, until=until, environments=list(environments)
    ):
        if history.subject(score)[1] in eval_traces:
            continue
        at = score.timestamp
        if score.name in SCORES:
            feature, name, worse, scale = SCORES[score.name]
            track(feature, name, worse, "quality", scale=scale).add(at, score.value)
        for value, feature, name, worse in SHARES.get(score.name, []):
            track(feature, name, worse, "quality", scale="absolute").add(at, score.value == value)

    return sorted(series.values(), key=lambda s: (s.feature, s.kind, s.name)), dict(changes)


def _number(value: float | None, unit: str, scale: str) -> str:
    if value is None:
        return "-"
    if unit == "$":
        return f"${value:.4f}" if value >= 0.001 else f"${value:.6f}"
    if scale == "absolute":
        return f"{value:.0%}"
    if unit == "s":
        return f"{value:.2f} s"
    return f"{value:,.1f}" if value < 100 else f"{value:,.0f}"


def _change(finding: Finding) -> str:
    if finding.change is None:
        return "-"
    points = finding.scale == "absolute"
    text = f"{finding.change * 100:+.0f}"
    if text in ("+0", "-0"):
        text = "0"  # a change too small to show is no change, not "-0%"
    return f"{text} pts" if points else f"{text}%"


def report(
    findings: list[Finding],
    changes: dict[str, Changes],
    since: datetime,
    split: datetime,
    until: datetime,
    environments: tuple[str, ...] = ENVIRONMENTS,
) -> str:
    """The comparison as Markdown: what changed, what drifted, what can't be told yet."""
    lines = [
        f"# Drift report, {until:%Y-%m-%d}",
        "",
        f"Recent: {split:%b %d} to {until:%b %d}. Baseline: {since:%b %d} to {split:%b %d}. "
        f"Environments: {', '.join(environments)}.",
        "",
        "## What changed between the windows",
    ]
    new = [
        f"- {feature}: {line}"
        for feature, seen in sorted(changes.items())
        for line in seen.new_since(split)
    ]
    lines += new or ["No new model, prompt version or release."]

    order = {"worse": 0, "changed": 1, "better": 2, "watch": 3}
    flagged = sorted((f for f in findings if f.verdict in order), key=lambda f: order[f.verdict])
    lines += ["", "## Drift"]
    if flagged:
        lines += _table(flagged)
    else:
        lines.append("Nothing moved by more than chance would explain.")

    steady = [f for f in findings if f.verdict == "steady"]
    if steady:
        lines += ["", "## Steady"] + _table(steady)

    thin = [f for f in findings if f.verdict == "not enough data"]
    if thin:
        lines += ["", f"## Not enough data yet (fewer than {MIN_VALUES} values in a window)"]
        for f in thin:
            lines.append(
                f"- {f.feature}: {f.name} ({f.baseline_n} in the baseline, {f.recent_n} recent)"
            )

    lines += [
        "",
        f"A metric is flagged when it moved by {USE_CHANGE:.0%} or more (10 points for a pass "
        "rate) and a permutation test, corrected for the number of metrics tested, puts the "
        f"odds of a false alarm under {FALSE_ALARM_RATE:.0%}. 'watch' is a move that large on "
        "too little data to rule out chance.",
    ]
    return "\n".join(lines)


def _p(value: float | None) -> str:
    if value is None:
        return "-"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _table(findings: list[Finding]) -> list[str]:
    rows = [
        "| Feature | Metric | Baseline | Recent | Change | p | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in findings:
        middle = " (median)" if f.stat == "median" else ""
        rows.append(
            f"| {f.feature} | {f.name}{middle} | "
            f"{_number(f.baseline, f.unit, f.scale)} (n={f.baseline_n}) | "
            f"{_number(f.recent, f.unit, f.scale)} (n={f.recent_n}) | {_change(f)} | "
            f"{_p(f.p_value)} | {f.verdict} |"
        )
    return rows
