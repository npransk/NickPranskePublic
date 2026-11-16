# Showcase copy of services/api/tests/test_tracing_e2e.py from the private mahjong-companion repo. Here to be read, not run.

"""What Langfuse actually receives for one strategist ask, with the real Anthropic and Langfuse
SDKs talking to a local stand-in server (``tracing_probe.py``). No network, no keys, no cost."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).parent / "tracing_probe.py"
SAMPLE_CARD = Path(__file__).parents[3] / "packages" / "engine" / "tests" / "sample_card.toml"
ANTHROPIC_SDK = "com.anthropic.sdk.python"


def probe(*flags):
    result = subprocess.run(
        [sys.executable, str(PROBE), str(SAMPLE_CARD), *flags],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def sent():
    return probe()


def test_each_claude_call_reaches_langfuse_once(sent):
    assert sent["pick"] == "1999-LIKE-1"
    assert sent["claude_requests"] == ["claude-sonnet-5-5"]
    generations = [span for span in sent["spans"] if span["type"] == "generation"]
    assert [g["name"] for g in generations] == ["claude"]
    assert {span["scope"] for span in sent["spans"]} == {"langfuse-sdk"}


def test_the_filter_is_what_removes_the_sdks_duplicate():
    """Without the app's filter the Anthropic SDK's own span is exported too, and Langfuse
    counts the call (and its cost) twice."""
    unfiltered = probe("--unfiltered")
    scopes = [span["scope"] for span in unfiltered["spans"]]
    assert scopes.count(ANTHROPIC_SDK) == 1
    assert scopes.count("langfuse-sdk") == 2


def test_every_span_carries_the_environment_game_feature_and_prompt_version(sent):
    for span in sent["spans"]:
        assert span["environment"] == "selftest"
        assert span["session"] == "probe-game"
        assert span["tags"] == ["strategist"]
        assert len(span["version"]) == 8
    assert {span["name"]: span["type"] for span in sent["spans"]} == {
        "strategist": "agent",
        "claude": "generation",
    }


def test_the_process_is_named_and_released(sent):
    assert sent["resource"]["service.name"] == "mahjong-api"
    assert sent["resource"]["langfuse.environment"] == "selftest"
    assert sent["resource"]["langfuse.release"] == "abc1234"


def test_the_asks_scores_are_sent(sent):
    assert sent["scores"] == {
        "strategist_outcome": "picked",
        "strategist_llm_calls": 1,
        "strategist_tool_calls": 0,
        "strategist_rejected_picks": 0,
        "pick_engine_rank": 1,
        "reason_words": 7,
        "reason_within_limit": 1,
    }
