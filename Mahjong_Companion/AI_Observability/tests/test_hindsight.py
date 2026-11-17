# Showcase copy of packages/engine/tests/test_hindsight.py from the private mahjong-companion repo. Here to be read, not run.

"""The replay the game review stands on: what the engine would have done at each of your moves,
on the made-up sample card."""

import json
from pathlib import Path

import pytest
from mahjong_engine.card import load_card
from mahjong_engine.game import Game, advise
from mahjong_engine.hindsight import Pick, facts
from mahjong_engine.reads import read_opponent

HERE = Path(__file__).parent
# Four from 1999-LIKE-1 (FF 3333 3333 3333): one 3 bam, one 3 crak and two 3 dots short.
HAND = ["flower", "flower", "3 bam", "3 bam", "3 bam", "3 crak", "3 crak", "3 crak"]
HAND += ["3 dot", "3 dot", "7 crak", "9 bam", "6 dot"]
# One 3 dot from 1999-LIKE-1.
READY = ["flower", "flower", *["3 bam"] * 4, *["3 crak"] * 4, *["3 dot"] * 3]


@pytest.fixture
def card():
    return load_card(HERE / "sample_card.toml")


def game(tiles, *events, messy=True):
    played = Game(players=4, dealer=2, messy=messy)
    for event in [{"type": "deal", "tiles": tiles}, *events]:
        played.add(event)
    return played


def discard(player, tile):
    return {"type": "discard", "player": player, "tile": tile}


def test_a_winning_discard_you_let_go_is_a_missed_mahjong(card):
    seen = facts(card, game(READY, discard(2, "3 dot"), discard(3, "9 bam")))
    (chance,) = seen["call_chances"]
    assert chance["move"] == 2 and chance["tile"] == "3 dot"
    assert chance["engine_suggested"].startswith("call it for Mahjong with")
    assert chance["you"] == "let it go"
    assert seen["missed_mahjong"] == [
        {
            "move": 2,
            "how": "on Player 2's discard of 3 dot",
            "hand": "Sample Like Numbers, top line (1999-LIKE-1)",
        }
    ]
    assert seen["result"] == "not over"


def test_mahjong_declared_on_a_discard_is_a_win_with_the_completed_hand(card):
    seen = facts(card, game(READY, discard(2, "3 dot"), {"type": "mahjong", "player": 1}))
    assert seen["call_chances"][0]["you"] == "declared Mahjong"
    assert seen["missed_mahjong"] == []
    assert seen["result"] == "you won"
    assert seen["tiles_away_at_end"] == 0  # the called tile completes it
    assert "1999-LIKE-1" in seen["your_closest_at_end"]


def test_a_complete_hand_drawn_and_not_declared_is_a_missed_mahjong(card):
    seen = facts(
        card,
        game(
            READY,
            discard(2, "9 bam"),
            discard(3, "9 bam"),
            discard(4, "9 bam"),
            {"type": "draw", "tile": "3 dot"},
            discard(1, "flower"),
        ),
    )
    (missed,) = seen["missed_mahjong"]
    assert missed["move"] == 5 and missed["how"] == "on your own draw"


def test_a_discard_is_set_against_the_engines_choice(card):
    played = game(
        HAND,
        discard(2, "1 bam"),
        discard(3, "1 bam"),
        discard(4, "1 bam"),
        {"type": "draw", "tile": "5 dot"},
    )
    suggested = str(advise(card, played.state()).discard)
    assert suggested != "3 bam"
    played.add(discard(1, "3 bam"))  # breaks up the 3s the closest hand is built on
    (mine,) = facts(card, played)["your_discards"]
    assert mine["move"] == 6 and mine["you_discarded"] == "3 bam"
    assert mine["engine_suggested"] == suggested
    assert mine["tiles_away_after"] > mine["tiles_away_if_suggested"]


def test_a_discard_the_engine_agrees_with_has_nothing_to_compare(card):
    played = game(HAND, discard(2, "1 bam"), discard(3, "1 bam"), discard(4, "1 bam"))
    played.add({"type": "draw", "tile": "5 dot"})
    played.add(discard(1, str(advise(card, played.state()).discard)))
    (mine,) = facts(card, played)["your_discards"]
    assert mine["you_discarded"] == mine["engine_suggested"]
    assert "tiles_away_if_suggested" not in mine


def test_a_discard_an_opponents_likely_hand_needs_is_flagged(card):
    played = game(
        HAND,
        discard(2, "5 dot"),
        {"type": "call", "player": 3, "tiles": ["5 dot", "5 dot", "joker"]},
        discard(3, "1 bam"),
        discard(4, "1 bam"),
        {"type": "draw", "tile": "2 crak"},
    )
    careful = read_opponent(card, played.state(), 3).careful
    assert careful, "the sample position should leave Player 3 a tile you hold"
    played.add(discard(1, str(careful[0])))
    seen = facts(card, played, opponent_cards={2: card, 3: card, 4: card})
    assert "Player 3" in seen["your_discards"][0]["an_opponent_may_need_it"]
    # Without the opponents' cards there is nothing to read them by.
    assert "an_opponent_may_need_it" not in facts(card, played)["your_discards"][0]


def test_a_discard_that_was_called_says_so(card):
    played = game(
        HAND,
        discard(2, "1 bam"),
        discard(3, "1 bam"),
        discard(4, "1 bam"),
        {"type": "draw", "tile": "5 dot"},
        discard(1, "5 dot"),
        {"type": "call", "player": 3, "tiles": ["5 dot", "joker", "joker"]},
    )
    assert facts(card, played)["your_discards"][0]["then"] == "Player 3 called it"


def test_charleston_passes_are_set_against_the_engines(card):
    played = game(
        HAND,
        {
            "type": "pass",
            "out": ["7 crak", "9 bam", "6 dot"],
            "received": ["1 bam", "2 bam", "4 dot"],
        },
        messy=False,
    )
    (passed,) = facts(card, played)["charleston_passes"]
    assert passed["direction"] == "right"
    assert passed["you_passed"] == ["7 crak", "9 bam", "6 dot"]
    assert len(passed["engine_would_pass"]) == 3
    expected = len(set(passed["you_passed"]) & set(passed["engine_would_pass"]))
    assert passed["same_tiles"] == expected
    assert passed["tiles_away_before"] == 4


def test_a_strategist_pick_is_followed_through_the_rest_of_the_game(card):
    played = game(
        HAND,
        discard(2, "3 dot"),
        discard(3, "3 bam"),
        {"type": "wall"},
    )
    picks = [Pick(1, 1, "1999-LIKE-1", True), Pick(2, 9, "1999-RUN-1")]
    seen = facts(card, played, picks)
    first, undone = seen["strategist_picks"]
    assert first["ask"] == 1 and first["move"] == 1 and first["you_followed"] is True
    assert first["engine_rank_then"] == 1 and first["tiles_away_then"] == 4
    assert first["tiles_it_needed"] == ["3 bam", "3 crak", "3 dot", "3 dot"]
    assert first["its_tiles_seen_later"] == [
        "3 dot discarded by Player 2",
        "3 bam discarded by Player 3",
    ]
    assert "tiles_away_at_end" in first and "still_possible_at_end" in first
    assert undone == {
        "ask": 2,
        "move": 9,
        "hand_id": "1999-RUN-1",
        "note": "asked at a point in the game that was later undone",
    }
    assert seen["result"] == "wall game: nobody won"


def test_the_facts_are_plain_data_and_follow_your_progress(card):
    played = game(HAND, discard(2, "1 bam"), discard(3, "1 bam"), discard(4, "1 bam"))
    played.add({"type": "draw", "tile": "5 dot"})
    played.add(discard(1, "5 dot"))
    seen = facts(card, played, [Pick(1, 4, "1999-LIKE-1")], {3: card})
    json.dumps(seen)
    assert [(p["move"], p["after"]) for p in seen["your_progress"]] == [
        (1, "deal"),
        (6, "discarded 5 dot"),
    ]
    assert seen["players"] == 4 and seen["moves"] == 6 and seen["charleston"] is False
