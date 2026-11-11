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


@pytest.fixture(autouse=True)
def no_langfuse(monkeypatch):
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY_MJ",
        "LANGFUSE_SECRET_KEY_MJ",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    strategist.langfuse_client.cache_clear()


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
