# Showcase copy of services/api/tests/test_drift.py from the private mahjong-companion repo. Here to be read, not run.

"""The drift report: when a change counts, and what is read from Langfuse to find one."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from mahjong_api import drift

SINCE = datetime(2026, 9, 1, tzinfo=UTC)
SPLIT = datetime(2026, 11, 1, tzinfo=UTC)
UNTIL = datetime(2026, 11, 15, tzinfo=UTC)
BEFORE = SPLIT - timedelta(days=10)
AFTER = SPLIT + timedelta(days=5)


def series(
    before, after, worse="up", kind="use", scale="relative", stat="mean", name="metric", unit=""
):
    s = drift.Series("strategist", name, worse, kind, unit=unit, scale=scale, stat=stat)
    for value in before:
        s.add(BEFORE, value)
    for value in after:
        s.add(AFTER, value)
    return s


def verdict(*args, **kwargs):
    (finding,) = drift.compare([series(*args, **kwargs)], SPLIT, SINCE)
    return finding


def test_a_gap_chance_cannot_explain_has_a_small_p_value_and_identical_windows_do_not():
    steady = [1000.0, 1010.0, 990.0, 1005.0, 995.0, 1002.0, 998.0, 1001.0]
    assert drift.permutation_p(steady, [v * 1.5 for v in steady]) < 0.01
    assert drift.permutation_p(steady, list(reversed(steady))) == 1.0
    assert drift.permutation_p(steady, steady[:4] + [1003.0, 997.0, 1004.0, 996.0]) > 0.5
    # The same inputs always give the same answer: a report can be rerun and compared.
    assert drift.permutation_p(steady, [1100.0] * 8) == drift.permutation_p(steady, [1100.0] * 8)


def test_testing_many_metrics_at_once_raises_the_bar_for_each():
    rates = drift.adjusted([0.01, 0.04, 0.03, 0.5])
    assert rates == pytest.approx([0.04, 0.0533, 0.0533, 0.5], abs=1e-3)
    assert drift.adjusted([]) == []
    assert drift.adjusted([0.2]) == [0.2]


def test_a_large_rise_in_tokens_is_worse_and_a_large_fall_is_better():
    low, high = [1000.0 + i for i in range(12)], [1400.0 + i for i in range(12)]
    worse = verdict(low, high)
    assert worse.verdict == "worse"
    assert worse.change == pytest.approx(0.4, abs=0.01)
    assert (worse.baseline_n, worse.recent_n) == (12, 12)
    assert verdict(high, low).verdict == "better"


def test_quality_scores_count_a_fall_as_worse_in_points_not_percent():
    followed = [1.0] * 18 + [0.0] * 2
    ignored = [1.0] * 8 + [0.0] * 12
    finding = verdict(followed, ignored, worse="down", kind="quality", scale="absolute")
    assert finding.verdict == "worse"
    assert finding.change == pytest.approx(-0.5)
    assert verdict(ignored, followed, worse="down", scale="absolute").verdict == "better"


def test_a_metric_with_no_good_direction_is_reported_as_changed():
    short, long = [12.0 + i % 3 for i in range(12)], [22.0 + i % 3 for i in range(12)]
    assert verdict(short, long, worse=None).verdict == "changed"


def test_small_moves_are_steady_however_sure():
    base = [1000.0 + i for i in range(40)]
    assert verdict(base, [v * 1.05 for v in base]).verdict == "steady"


def test_a_big_move_on_little_data_is_only_watched():
    assert (
        verdict(
            [1000.0, 1300.0, 900.0, 1250.0, 950.0], [1500.0, 900.0, 1600.0, 1000.0, 1700.0]
        ).verdict
        == "watch"
    )


def test_fewer_than_five_values_in_a_window_is_not_enough_to_say():
    finding = verdict([1000.0] * 30, [5000.0] * 4)
    assert finding.verdict == "not enough data" and finding.p_value is None
    assert finding.recent == 5000.0  # still shown, just not judged
    assert verdict([], [1.0] * 9).verdict == "not enough data"


def test_latency_is_compared_by_its_median_so_one_slow_call_is_not_drift():
    usual = [2.0 + i * 0.01 for i in range(10)]
    finding = verdict(usual, usual[:9] + [60.0], stat="median")
    assert finding.verdict == "steady" and finding.stat == "median"


def test_values_outside_the_two_windows_are_ignored():
    s = series([1000.0] * 6, [1000.0] * 6)
    s.add(SINCE - timedelta(days=1), 99999.0)
    (finding,) = drift.compare([s], SPLIT, SINCE)
    assert (finding.baseline_n, finding.baseline) == (6, 1000.0)


def generation(name, at, trace="t", trace_name=None, session="game-1", **more):
    fields = {
        "id": f"{trace}-{name}-{at.isoformat()}",
        "trace_id": trace,
        "name": name,
        "type": "GENERATION",
        "trace_name": trace_name or name,
        "session_id": session,
        "environment": "production",
        "start_time": at,
        "level": "DEFAULT",
        "model": "claude-sonnet-5-5",
        "version": "aaaa1111",
        "release": "abc1234",
        "metadata": {},
        "usage_details": {"input": 100, "output": 20},
        "total_cost": 0.01,
        "latency": 2.0,
    }
    return SimpleNamespace(**{**fields, **more})


class Archive:
    """Stands in for Langfuse's read API (one page is enough here)."""

    def __init__(self, observations=(), scores=()):
        self._observations, self._scores = list(observations), list(scores)
        self.asked = []
        self.api = SimpleNamespace(
            observations=SimpleNamespace(get_many=self._observations_page),
            scores=SimpleNamespace(get_many=self._scores_page),
        )

    def _observations_page(self, type=None, name=None, **filters):
        self.asked.append({"type": type, "name": name, **filters})
        rows = [
            o for o in self._observations if o.type == type and (name is None or o.name == name)
        ]
        return SimpleNamespace(data=rows, meta=SimpleNamespace(cursor=None))

    def _scores_page(self, **filters):
        return SimpleNamespace(data=self._scores, meta=SimpleNamespace(cursor=None))


def by_name(all_series):
    return {(s.feature, s.name): [value for _, value in s.points] for s in all_series}


def test_generations_become_per_call_and_per_ask_series():
    archive = Archive(
        [
            # One strategist ask of two calls; the second reads the cache the first wrote.
            generation(
                "claude",
                BEFORE,
                trace="ask-1",
                trace_name="strategist",
                usage_details={"input": 4, "output": 50, "cache_creation_input_tokens": 1800},
                total_cost=0.006,
            ),
            generation(
                "claude",
                BEFORE + timedelta(seconds=3),
                trace="ask-1",
                trace_name="strategist",
                usage_details={"input": 300, "output": 140, "cache_read_input_tokens": 1800},
                total_cost=0.003,
                latency=3.0,
            ),
            generation(
                "read-hand", BEFORE, trace="photo-1", usage_details={"input": 3400, "output": 240}
            ),
            generation("claude", BEFORE, trace="draft-1", trace_name="draft-card"),
            generation("claude-round-2", BEFORE, trace="draft-1", trace_name="draft-card"),
            generation("jev", BEFORE, trace="line-1", trace_name="heard", total_cost=0.00005),
        ]
    )
    found = by_name(drift.collect(archive, SINCE, UNTIL)[0])
    assert found[("strategist", "prompt tokens per call")] == [1804.0, 2100.0]
    assert found[("strategist", "output tokens per call")] == [50.0, 140.0]
    assert found[("strategist", "Claude calls per ask")] == [2.0]
    assert found[("strategist", "tokens per ask")] == [4094.0]
    assert found[("strategist", "cost per ask")] == [pytest.approx(0.009)]
    assert found[("strategist", "later calls served from cache")] == [1.0]
    assert found[("photo reading", "prompt tokens per call")] == [3400.0]
    assert len(found[("card drafting", "cost per call")]) == 2  # old and new round names
    assert found[("table talk", "cost per call")] == [0.00005]
    # Only real games are read, and without inputs or outputs (they aren't needed).
    assert archive.asked[0]["environment"] == ["production", "default"]
    assert "io" not in archive.asked[0]["fields"]


def test_failed_calls_count_as_failures_not_as_cheap_calls():
    archive = Archive(
        [
            generation("read-hand", BEFORE, trace="a"),
            generation(
                "read-hand", BEFORE, trace="b", level="ERROR", usage_details={}, total_cost=None
            ),
        ]
    )
    found = by_name(drift.collect(archive, SINCE, UNTIL)[0])
    assert found[("photo reading", "calls that failed")] == [0.0, 1.0]
    assert found[("photo reading", "prompt tokens per call")] == [100.0]


def score(name, value, at=BEFORE, trace="t"):
    return SimpleNamespace(
        name=name, value=value, timestamp=at, subject={"kind": "trace", "id": trace}
    )


def test_scores_become_quality_series_and_categories_become_shares():
    archive = Archive(
        scores=[
            score("recommendation_followed", True),
            score("recommendation_followed", False),
            score("strategist_hindsight", "poor"),
            score("tile_fixes", 2),
            score("heard_outcome", "auto"),
            score("heard_outcome", "fixed"),
            score("strategist_outcome", "refused"),
            score("some_other_score", 1),
        ]
    )
    all_series, _ = drift.collect(archive, SINCE, UNTIL)
    found = by_name(all_series)
    assert found[("strategist", "picks the player followed")] == [1.0, 0.0]
    assert found[("strategist", "picks the game review called poor")] == [1.0]
    assert found[("photo reading", "fixes per photo")] == [2.0]
    assert found[("table talk", "heard lines logged without asking")] == [1.0, 0.0]
    assert found[("table talk", "heard lines the player had to fix")] == [0.0, 1.0]
    assert found[("strategist", "asks that ended in a pick")] == [0.0]
    assert len(found) == 6  # the unknown score is left out
    directions = {(s.feature, s.name): (s.worse, s.kind, s.scale) for s in all_series}
    assert directions[("strategist", "picks the player followed")] == (
        "down",
        "quality",
        "absolute",
    )
    assert directions[("photo reading", "fixes per photo")] == ("up", "quality", "relative")


def test_runs_recorded_before_evals_had_their_own_environment_are_left_out():
    archive = Archive(
        [
            generation("read-hand", BEFORE, trace="eval-photo", session="vision-eval"),
            generation("read-hand", BEFORE, trace="real-photo"),
            generation("anthropic.messages.create", BEFORE, trace="real-photo"),  # the SDK's copy
        ],
        scores=[
            score("tile_cv_correct", 1.0, trace="eval-photo"),
            score("tile_cv_correct", 0.9, trace="real-photo"),
        ],
    )
    found = by_name(drift.collect(archive, SINCE, UNTIL)[0])
    assert found[("photo reading", "cost per call")] == [0.01]
    assert found[("photo reading", "tiles read right")] == [0.9]


def test_heard_lines_sent_to_jev_are_tracked_from_old_and_new_traces():
    def heard(metadata):
        return SimpleNamespace(
            type="CHAIN", name="heard", start_time=BEFORE, metadata=metadata, session_id="game-1"
        )

    archive = Archive(
        [
            heard({"route": "grammar", "decided_by": "grammar"}),
            heard({"route": "jev_failed", "decided_by": "grammar"}),  # asked, though Jev was down
            heard({"decided_by": "jev"}),  # recorded before "route" existed
            heard({}),
        ]
    )
    found = by_name(drift.collect(archive, SINCE, UNTIL)[0])
    assert found[("table talk", "heard lines sent to Jev")] == [0.0, 1.0, 1.0]


def test_new_models_prompts_and_releases_are_listed_as_the_likely_cause():
    archive = Archive(
        [
            generation("claude", BEFORE, trace="a", trace_name="strategist"),
            generation(
                "claude",
                AFTER,
                trace="b",
                trace_name="strategist",
                version="bbbb2222",
                release="def5678",
                metadata={"served_model": "claude-sonnet-5-5-20261101"},
            ),
            generation("read-hand", BEFORE, trace="c"),
            generation("read-hand", AFTER, trace="d"),
        ]
    )
    _, changes = drift.collect(archive, SINCE, UNTIL)
    assert changes["strategist"].new_since(SPLIT) == [
        "model claude-sonnet-5-5-20261101 first seen 2026-11-06 (was claude-sonnet-5-5)",
        "prompt bbbb2222 first seen 2026-11-06 (was aaaa1111)",
        "release def5678 first seen 2026-11-06 (was abc1234)",
    ]
    assert changes["photo reading"].new_since(SPLIT) == []


def test_the_report_leads_with_causes_then_drift_then_what_cannot_be_told():
    low, high = [1000.0 + i for i in range(12)], [1400.0 + i for i in range(12)]
    findings = drift.compare(
        [
            series(low, high, name="prompt tokens per call"),
            series([2.0] * 9, [2.0] * 9, name="latency per call", stat="median", unit="s"),
            series(
                [1.0], [], worse="down", kind="quality", scale="absolute", name="picks followed"
            ),
        ],
        SPLIT,
        SINCE,
    )
    changes = {"strategist": drift.Changes()}
    changes["strategist"].note("prompt", "aaaa1111", BEFORE)
    changes["strategist"].note("prompt", "bbbb2222", AFTER)
    text = drift.report(findings, changes, SINCE, SPLIT, UNTIL)
    assert text.index("## What changed") < text.index("## Drift") < text.index("## Steady")
    assert "- strategist: prompt bbbb2222 first seen 2026-11-06 (was aaaa1111)" in text
    assert (
        "| strategist | prompt tokens per call | 1,006 (n=12) | 1,406 (n=12) | +40% | <0.001 | "
        "worse |" in text
    )
    assert "| strategist | latency per call (median) | 2.00 s (n=9) | 2.00 s (n=9) | 0% |" in text
    assert "- strategist: picks followed (1 in the baseline, 0 recent)" in text


def test_a_quiet_report_says_so():
    text = drift.report([], {}, SINCE, SPLIT, UNTIL)
    assert "No new model, prompt version or release." in text
    assert "Nothing moved by more than chance would explain." in text
