# Showcase copy of services/api/tests/test_strategist.py from the private mahjong-companion repo. Here to be read, not run.

"""Strategist tests with a scripted stand-in for Claude: no API key, no network, no cost."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from mahjong_api import games, strategist
from mahjong_api.main import app
from mahjong_api.store import LocalStore
from mahjong_engine.card import load_card
from mahjong_engine.game import Game

SAMPLE_CARD = Path(__file__).parents[3] / "packages" / "engine" / "tests" / "sample_card.toml"
HAND = ["flower", "flower", "3 bam", "3 bam", "3 bam", "3 crak", "3 crak", "3 crak"]
HAND += ["3 dot", "3 dot", "7 crak", "9 bam", "6 dot"]


def usage():
    return SimpleNamespace(
        input_tokens=1000,
        output_tokens=50,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def tool_call(name, **tool_input):
    block = SimpleNamespace(type="tool_use", id=f"call-{name}", name=name, input=tool_input)
    return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage())


def text(message):
    block = SimpleNamespace(type="text", text=message)
    return SimpleNamespace(content=[block], stop_reason="end_turn", usage=usage())


class FakeClaude:
    """Replays scripted responses and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        # Snapshot the messages: the real client sends them before the list grows again.
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def state():
    game = Game(players=2, dealer=2)
    game.add({"type": "deal", "tiles": HAND})
    return game.state()


@pytest.fixture
def card():
    return load_card(SAMPLE_CARD)


def test_looks_closer_then_picks(card, state):
    claude = FakeClaude(
        tool_call("opponent_reads"),
        tool_call("hand_details", hand_id="1999-LIKE-1"),
        tool_call("submit_pick", hand_id="1999-LIKE-1", reason="Go for like 3s: you hold nine."),
    )
    pick = strategist.ask(claude, card, state, "game-1")
    assert pick.hand_id == "1999-LIKE-1"
    assert pick.label == "Sample Like Numbers, top line"
    assert len(claude.requests) == 3
    first = claude.requests[0]
    assert first["model"] == strategist.MODEL
    assert first["output_config"] == {"effort": "low"}
    assert "1999-LIKE-1" in first["messages"][0]["content"]
    # Tool results go back in the next request, matched to their calls.
    result = claude.requests[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "call-opponent_reads"
    assert "Player 2" in result["content"]


def test_a_hand_outside_the_candidates_is_rejected(card, state):
    claude = FakeClaude(
        tool_call("submit_pick", hand_id="2024-MADE-UP-9", reason="..."),
        tool_call("submit_pick", hand_id="1999-LIKE-1", reason="Closest by far."),
    )
    pick = strategist.ask(claude, card, state, "game-1")
    assert pick.hand_id == "1999-LIKE-1"
    rejection = claude.requests[1]["messages"][-1]["content"][0]
    assert rejection["is_error"] is True


def test_a_text_answer_is_nudged_toward_the_tool(card, state):
    claude = FakeClaude(
        text("I'd go for like numbers."),
        tool_call("submit_pick", hand_id="1999-LIKE-1", reason="Closest by far."),
    )
    assert strategist.ask(claude, card, state, "game-1").hand_id == "1999-LIKE-1"
    assert "submit_pick" in claude.requests[1]["messages"][-1]["content"][0]["text"]


def test_gives_up_after_the_call_limit(card, state):
    claude = FakeClaude(*[tool_call("opponent_reads")] * strategist.MAX_CALLS)
    with pytest.raises(strategist.StrategistError):
        strategist.ask(claude, card, state, "game-1")


def test_a_refusal_is_an_error(card, state):
    refusal = SimpleNamespace(content=[], stop_reason="refusal", usage=usage())
    with pytest.raises(strategist.StrategistError, match="declined"):
        strategist.ask(FakeClaude(refusal), card, state, "game-1")


def test_an_ask_is_traced_with_everything_it_was_based_on(card, state, langfuse):
    claude = FakeClaude(
        tool_call("opponent_reads"),
        tool_call("hand_details", hand_id="1999-LIKE-1"),
        tool_call("submit_pick", hand_id="1999-LIKE-1", reason="Go for like 3s: you hold nine."),
    )
    pick = strategist.ask(claude, card, state, "game-1")
    assert pick.trace_id == langfuse.trace_id
    (agent,) = langfuse.named("strategist")
    # The situation and the engine's numbers are on the trace, so a judge needs nothing else.
    assert "Your tiles (13)" in agent["input"]["situation"]
    assert agent["input"]["candidates"][0]["hand_id"] == "1999-LIKE-1"
    assert agent["output"] == {
        "hand_id": "1999-LIKE-1",
        "label": "Sample Like Numbers, top line",
        "reason": "Go for like 3s: you hold nine.",
        "engine_rank": 1,
    }
    assert agent["metadata"] == {
        "llm_calls": 3,
        "tool_calls": 2,
        "rejected_picks": 0,
        "nudges": 0,
        "outcome": "picked",
    }
    calls = langfuse.named("claude")
    assert [c["metadata"]["round"] for c in calls] == [1, 2, 3]
    assert calls[0]["input"][0] == {"role": "system", "content": strategist.SYSTEM}
    assert calls[0]["metadata"]["stop_reason"] == "tool_use"
    assert calls[0]["usage_details"]["input"] == 1000
    assert [o["name"] for o in langfuse.observations if o["as_type"] == "tool"] == [
        "opponent_reads",
        "hand_details",
    ]
    assert langfuse.flushed == 1


def test_an_ask_is_scored_by_checks_that_need_no_judgment(card, state, langfuse):
    claude = FakeClaude(
        tool_call("opponent_reads"),
        tool_call("submit_pick", hand_id="2024-MADE-UP-9", reason="..."),
        tool_call("submit_pick", hand_id="1999-LIKE-1", reason="Go for like 3s: you hold nine."),
    )
    strategist.ask(claude, card, state, "game-1")
    assert langfuse.score("strategist_outcome") == "picked"
    assert langfuse.score("pick_engine_rank") == 1
    assert langfuse.score("reason_words") == 7
    assert langfuse.score("reason_within_limit") == 1.0
    assert langfuse.score("strategist_llm_calls") == 3
    assert langfuse.score("strategist_tool_calls") == 1
    assert langfuse.score("strategist_rejected_picks") == 1  # the made-up hand
    # One score of each kind per ask: asking Langfuse again would overwrite, not add.
    ids = [s["score_id"] for s in langfuse.scores]
    assert len(ids) == len(set(ids)) == 7
    assert all(s["trace_id"] == langfuse.trace_id for s in langfuse.scores)
    rejected = [o for o in langfuse.named("submit_pick")]
    assert rejected[0]["level"] == "WARNING"


def test_a_reason_over_the_word_limit_is_flagged(card, state, langfuse):
    rambling = " ".join(["word"] * (strategist.REASON_WORDS + 1))
    claude = FakeClaude(tool_call("submit_pick", hand_id="1999-RUN-1", reason=rambling))
    strategist.ask(claude, card, state, "game-1")
    assert langfuse.score("reason_within_limit") == 0.0
    assert langfuse.score("reason_words") == strategist.REASON_WORDS + 1
    assert langfuse.score("pick_engine_rank") > 1  # not the engine's closest hand


@pytest.mark.parametrize(
    "responses, outcome",
    [
        ([SimpleNamespace(content=[], stop_reason="refusal", usage=usage())], "refused"),
        ([tool_call("opponent_reads")] * strategist.MAX_CALLS, "no_pick"),
    ],
)
def test_an_ask_without_a_pick_is_scored_and_marked_as_an_error(
    card, state, langfuse, responses, outcome
):
    with pytest.raises(strategist.StrategistError):
        strategist.ask(FakeClaude(*responses), card, state, "game-1")
    assert langfuse.score("strategist_outcome") == outcome
    assert "pick_engine_rank" not in langfuse.score_names()
    (agent,) = langfuse.named("strategist")
    assert agent["level"] == "ERROR"
    assert agent["metadata"]["outcome"] == outcome
    assert langfuse.flushed == 1


def test_the_api_failing_is_scored_as_an_error_and_still_raised(card, state, langfuse):
    class Down:
        def __init__(self):
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

        def _create(self, **request):
            raise ConnectionError("no route to the API")

    with pytest.raises(ConnectionError):
        strategist.ask(Down(), card, state, "game-1")
    assert langfuse.score("strategist_outcome") == "error"
    assert langfuse.score("strategist_llm_calls") == 1
    assert langfuse.named("claude")[0]["level"] == "ERROR"
    assert "ConnectionError" in langfuse.named("strategist")[0]["status_message"]


def test_the_prompt_version_changes_with_the_prompt():
    before = strategist.PROMPT_VERSION
    assert before == strategist.prompt_version(strategist.SYSTEM, strategist.TOOLS)
    assert before != strategist.prompt_version(strategist.SYSTEM + " Be brief.", strategist.TOOLS)


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "cards").mkdir()
    shutil.copy(SAMPLE_CARD, tmp_path / "cards" / "2024.toml")
    monkeypatch.setattr(games, "_cards", {})
    app.dependency_overrides[games.get_store] = lambda: LocalStore(tmp_path)
    yield TestClient(app)
    app.dependency_overrides.clear()


def started_game(client):
    game_id = client.post("/api/games", json={"players": 2, "dealer": 2}).json()["id"]
    client.post(f"/api/games/{game_id}/events", json={"type": "deal", "tiles": HAND})
    return game_id


def test_route_returns_the_pick(client):
    game_id = started_game(client)
    claude = FakeClaude(tool_call("submit_pick", hand_id="1999-LIKE-1", reason="Closest."))
    app.dependency_overrides[games.get_claude] = lambda: claude
    pick = client.post(f"/api/games/{game_id}/strategist").json()
    assert pick["hand_id"] == "1999-LIKE-1"
    assert pick["reason"] == "Closest."


def test_route_without_an_api_key_is_unavailable(client):
    game_id = started_game(client)
    app.dependency_overrides[games.get_claude] = lambda: None
    assert client.post(f"/api/games/{game_id}/strategist").status_code == 503


def test_feedback_is_recorded(client):
    game_id = started_game(client)
    body = {"trace_id": "0" * 32, "hand_id": "1999-LIKE-1", "followed": True}
    assert client.post(f"/api/games/{game_id}/strategist/feedback", json=body).json() == {
        "recorded": True
    }


def test_feedback_is_one_score_per_ask_that_a_change_of_mind_overwrites(client, langfuse):
    game_id = started_game(client)
    for followed in (True, False):
        body = {"trace_id": "0" * 32, "hand_id": "1999-LIKE-1", "followed": followed}
        client.post(f"/api/games/{game_id}/strategist/feedback", json=body)
    first, second = langfuse.scores
    assert (first["name"], first["data_type"]) == ("recommendation_followed", "BOOLEAN")
    assert [first["value"], second["value"]] == [1.0, 0.0]
    assert first["score_id"] == second["score_id"]
    assert first["metadata"] == {"game": game_id, "hand_id": "1999-LIKE-1"}
