# Showcase copy of services/api/tests/test_review.py from the private mahjong-companion repo. Here to be read, not run.

"""Score My Game with a scripted stand-in for Claude: no API key, no network, no cost."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from mahjong_api import games, review
from mahjong_api.main import app
from mahjong_api.store import LocalStore

SAMPLE_CARD = Path(__file__).parents[3] / "packages" / "engine" / "tests" / "sample_card.toml"
HAND = ["flower", "flower", "3 bam", "3 bam", "3 bam", "3 crak", "3 crak", "3 crak"]
HAND += ["3 dot", "3 dot", "7 crak", "9 bam", "6 dot"]
GAME = [
    {"type": "discard", "player": 2, "tile": "1 bam"},
    {"type": "draw", "tile": "5 dot"},
    {"type": "discard", "player": 1, "tile": "3 bam"},
    {"type": "discard", "player": 2, "tile": "2 bam"},
]


def usage():
    return SimpleNamespace(
        input_tokens=3000,
        output_tokens=900,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def answer(**changes):
    reply = {
        "play_score": 4,
        "summary": "A short game you lost on the draw.",
        "went_well": ["Kept your flowers.", "Asked early.", "Watched the discards.", "Extra."],
        "moments": [
            {
                "move": 4,
                "kind": "mistake",
                "what_happened": "You let a 3 bam go.",
                "better": "Let the 5 dot go.",
                "why": "The 3s were your closest hand.",
            },
            {"move": 99, "kind": "good", "what_happened": "x", "better": "", "why": "x"},
        ],
        "strategist": [
            {"ask": 1, "verdict": "good", "why": "Like 3s was four away and the clear best."},
            {"ask": 7, "verdict": "poor", "why": "No such ask."},
        ],
        "strategist_summary": "Sound advice.",
        "work_on": "Check the suggested discard before letting a set go.",
    }
    return json.dumps(reply | changes)


def reply(text, stop_reason="end_turn", model="claude-opus-5-5"):
    blocks = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=text),
    ]
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=usage(), model=model)


class Stream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeClaude:
    """Streams scripted reviews and answers strategist asks with the engine's closest hand."""

    def __init__(self, *reviews):
        self.reviews = list(reviews)
        self.requests = []
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(stream=self._stream, create=self._create)
        )

    def _stream(self, **request):
        self.requests.append(request)
        return Stream(self.reviews.pop(0))

    def _create(self, **request):
        pick = {"hand_id": "1999-LIKE-1", "reason": "Go for like 3s: you hold nine."}
        block = SimpleNamespace(type="tool_use", id="call-1", name="submit_pick", input=pick)
        return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage())


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "cards").mkdir()
    shutil.copy(SAMPLE_CARD, tmp_path / "cards" / "2024.toml")
    monkeypatch.setattr(games, "_cards", {})
    app.dependency_overrides[games.get_store] = lambda: LocalStore(tmp_path)
    yield TestClient(app)
    app.dependency_overrides.clear()


def use(claude):
    app.dependency_overrides[games.get_claude] = lambda: claude
    return claude


def play(client, *events, ask_after_deal=False, ending=None):
    game_id = client.post("/api/games", json={"players": 2, "dealer": 2}).json()["id"]
    client.post(f"/api/games/{game_id}/events", json={"type": "deal", "tiles": HAND})
    if ask_after_deal:
        pick = client.post(f"/api/games/{game_id}/strategist").json()
        assert pick["ask"] == 1
    for event in [*events, *([ending] if ending else [])]:
        response = client.post(f"/api/games/{game_id}/events", json=event)
        assert response.status_code == 200, response.text
    return game_id


MAHJONG = {"type": "mahjong", "player": 2}


def test_a_finished_game_is_reviewed_and_kept_with_it(client, langfuse):
    claude = use(FakeClaude(reply(answer())))
    game_id = play(client, *GAME, ask_after_deal=True, ending=MAHJONG)
    client.post(
        f"/api/games/{game_id}/strategist/feedback",
        json={"hand_id": "1999-LIKE-1", "followed": True, "ask": 1},
    )
    response = client.post(f"/api/games/{game_id}/review")
    assert response.status_code == 200, response.text
    scored = response.json()["review"]
    assert scored["play_score"] == 4 and scored["moves"] == 6
    assert len(scored["went_well"]) == 3
    assert [m["move"] for m in scored["moments"]] == [4]  # move 99 doesn't exist
    (verdict,) = scored["strategist"]  # ask 7 doesn't exist
    assert verdict == {
        "ask": 1,
        "move": 1,
        "hand_id": "1999-LIKE-1",
        "label": "Sample Like Numbers, top line",
        "followed": True,
        "verdict": "good",
        "why": "Like 3s was four away and the clear best.",
    }
    assert scored["helpful"] is None and scored["trace_id"] == langfuse.trace_id
    # Kept: reopening the game shows it without asking Claude again.
    assert client.get(f"/api/games/{game_id}/review").json() == {"review": scored}
    assert len(claude.requests) == 1

    # Claude was told the engine's facts and what the strategist said, nothing else.
    (request,) = claude.requests
    assert request["model"] == "claude-opus-5-5"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["schema"] == review.SCHEMA
    assert request["fallbacks"] == "default"
    told = json.loads(request["messages"][0]["content"].split("\n", 1)[1])
    assert told["result"] == "Player 2 won"
    (discard,) = told["your_discards"]
    assert discard["move"] == 4 and discard["you_discarded"] == "3 bam"
    (pick,) = told["strategist_picks"]
    assert pick["strategist_said"] == "Go for like 3s: you hold nine."
    assert pick["you_followed"] is True


def test_a_review_is_traced_and_scored(client, langfuse):
    use(FakeClaude(reply(answer())))
    game_id = play(client, *GAME, ask_after_deal=True, ending=MAHJONG)
    langfuse.scores.clear()
    client.post(f"/api/games/{game_id}/review")
    (root,) = langfuse.named("review-game")
    assert root["as_type"] == "evaluator"
    assert root["output"]["play_score"] == 4
    assert root["metadata"] == {"dropped_refs": 2}  # move 99 and ask 7
    (replay,) = langfuse.named("hindsight")
    assert replay["output"]["your_discards"] == 1 and replay["output"]["strategist_picks"] == 1
    (generation,) = langfuse.named("review")
    assert generation["model"] == "claude-opus-5-5"
    assert generation["usage_details"]["output"] == 900
    scores = {s["name"]: s for s in langfuse.scores}
    assert scores["review_outcome"]["value"] == "reviewed"
    assert scores["review_dropped_refs"]["value"] == 2.0
    play_score = scores["review_play_score"]
    assert (play_score["value"], play_score["session_id"]) == (4.0, game_id)
    assert play_score["score_id"] == f"{game_id}-review_play_score"  # scoring again overwrites
    hindsight = scores["strategist_hindsight"]
    # On the strategist's own trace, so its dashboards show how its picks held up.
    assert hindsight["value"] == "good" and hindsight["trace_id"] == langfuse.trace_id
    assert hindsight["metadata"]["ask"] == 1
    assert langfuse.flushed >= 1


def test_only_a_finished_game_can_be_scored(client):
    use(FakeClaude())
    game_id = play(client, *GAME)
    response = client.post(f"/api/games/{game_id}/review")
    assert response.status_code == 400
    assert "over" in response.json()["detail"]


def test_a_wall_game_can_be_scored(client):
    use(FakeClaude(reply(answer(strategist=[]))))
    game_id = play(client, *GAME, ending={"type": "wall"})
    scored = client.post(f"/api/games/{game_id}/review").json()["review"]
    assert scored["strategist"] == [] and scored["strategist_summary"] == ""


def test_without_an_api_key_reviews_are_unavailable(client):
    game_id = play(client, *GAME, ending=MAHJONG)
    use(None)
    assert client.post(f"/api/games/{game_id}/review").status_code == 503


def test_a_refusal_is_reported_scored_and_not_kept(client, langfuse):
    use(FakeClaude(reply("", stop_reason="refusal")))
    game_id = play(client, *GAME, ending=MAHJONG)
    response = client.post(f"/api/games/{game_id}/review")
    assert response.status_code == 502
    assert "declined" in response.json()["detail"]
    assert langfuse.score("review_outcome") == "refused"
    assert langfuse.named("review-game")[0]["level"] == "ERROR"
    assert client.get(f"/api/games/{game_id}/review").json() == {"review": None}


@pytest.mark.parametrize(
    ("text", "problem"),
    [("not json", "unreadable"), (answer(play_score=0), "score of 0")],
)
def test_an_answer_that_cannot_be_used_is_an_error(client, langfuse, text, problem):
    use(FakeClaude(reply(text)))
    game_id = play(client, *GAME, ending=MAHJONG)
    response = client.post(f"/api/games/{game_id}/review")
    assert response.status_code == 502 and problem in response.json()["detail"]
    assert langfuse.score("review_outcome") == "error"
    assert "review_play_score" not in langfuse.score_names()


def test_scoring_again_replaces_the_review(client):
    use(FakeClaude(reply(answer()), reply(answer(play_score=7))))
    game_id = play(client, *GAME, ending=MAHJONG)
    client.post(f"/api/games/{game_id}/review")
    client.post(f"/api/games/{game_id}/review")
    assert client.get(f"/api/games/{game_id}/review").json()["review"]["play_score"] == 7


def test_the_players_verdict_on_a_review_is_kept_and_scored(client, langfuse):
    use(FakeClaude(reply(answer())))
    game_id = play(client, *GAME, ending=MAHJONG)
    url = f"/api/games/{game_id}/review/feedback"
    assert client.post(url, json={"useful": True}).status_code == 404  # not scored yet
    client.post(f"/api/games/{game_id}/review")
    for useful in (False, True):
        assert client.post(url, json={"useful": useful}).json() == {"recorded": True}
    assert client.get(f"/api/games/{game_id}/review").json()["review"]["helpful"] is True
    helpful = [s for s in langfuse.scores if s["name"] == "review_helpful"]
    assert [s["value"] for s in helpful] == [0.0, 1.0]
    assert {s["score_id"] for s in helpful} == {f"{langfuse.trace_id}-review_helpful"}


def test_kept_notes_are_not_listed_as_games(client):
    use(FakeClaude())
    game_id = play(client, ask_after_deal=True)
    assert [g["id"] for g in client.get("/api/games").json()] == [game_id]


def test_the_prompt_version_changes_with_the_prompt_or_the_answer_shape():
    assert review.PROMPT_VERSION == review.prompt_version(review.SYSTEM, review.SCHEMA)
    assert review.PROMPT_VERSION != review.prompt_version(
        review.SYSTEM + " Be kind.", review.SCHEMA
    )
