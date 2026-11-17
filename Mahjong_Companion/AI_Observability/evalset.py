# Showcase copy of services/api/src/mahjong_api/evalset.py from the private mahjong-companion repo. Here to be read, not run.

"""Fixed, made-up test material for measuring the AI parts over time (ADR 7).

Everything here is played on the made-up sample card (``packages/engine/tests``), never a real
NMJL card, so it can live in git and in Langfuse datasets.

- ``POSITIONS``: game positions the strategist is asked about again and again. The inputs never
  change, so a change in its answers, tokens or latency is the model or the prompt, not the games
  (``scripts/strategist_eval.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mahjong_engine.card import Card, load_card
from mahjong_engine.game import Game, GameState, advise
from mahjong_engine.odds import HandOdds
from mahjong_engine.reads import Read, read_opponent

SAMPLE_CARD = Path(__file__).parents[4] / "packages" / "engine" / "tests" / "sample_card.toml"
CANDIDATES = 5


def discard(player: int, tile: str) -> dict[str, Any]:
    return {"type": "discard", "player": player, "tile": tile}


@dataclass(frozen=True)
class Position:
    """A moment in a made-up game: who is playing, your tiles and what has happened since the
    deal (game events, as the app records them)."""

    name: str
    about: str
    players: int
    dealer: int
    tiles: tuple[str, ...]
    events: tuple[dict[str, Any], ...] = ()

    def game(self) -> Game:
        game = Game(players=self.players, dealer=self.dealer, messy=True)
        game.add({"type": "deal", "tiles": list(self.tiles)})
        for event in self.events:
            game.add(dict(event))
        return game

    def state(self) -> GameState:
        return self.game().state()


_POSITIONS = [
    Position(
        "triple-threes",
        "One hand is far closer than the rest: three sets of 3s already made.",
        players=2,
        dealer=2,
        tiles=(
            *("3 bam",) * 3,
            *("3 crak",) * 3,
            *("3 dot",) * 2,
            *("flower",) * 2,
            "7 crak",
            "9 bam",
            "6 dot",
        ),
    ),
    Position(
        "winds-and-fives",
        "The closest hand is concealed, and two of its 5 bams have already been discarded.",
        players=4,
        dealer=3,
        tiles=(
            "north wind",
            "east wind",
            "west wind",
            "south wind",
            "5 bam",
            "5 bam",
            "green dragon",
            "green dragon",
            "5 crak",
            "red dragon",
            "2 dot",
            "4 dot",
            "6 dot",
        ),
        events=(discard(3, "5 bam"), discard(4, "5 bam")),
    ),
    Position(
        "evens",
        "A concealed singles hand is 3 tiles away; the joker cannot help with singles.",
        players=3,
        dealer=2,
        tiles=(
            "2 bam",
            "4 bam",
            "6 bam",
            "8 bam",
            "2 crak",
            "4 crak",
            "6 crak",
            "2 dot",
            "4 dot",
            "9 dot",
            "flower",
            "flower",
            "joker",
        ),
        events=(discard(2, "8 dot"), discard(3, "8 dot")),
    ),
    Position(
        "dot-run",
        "2 tiles from a run, but both other red dragons it needs are already discarded.",
        players=4,
        dealer=2,
        tiles=(
            *("1 dot",) * 3,
            *("2 dot",) * 3,
            *("3 dot",) * 3,
            "red dragon",
            "red dragon",
            "joker",
            "7 bam",
        ),
        events=(discard(2, "red dragon"), discard(3, "red dragon"), discard(4, "flower")),
    ),
    Position(
        "joker-swap",
        "An opponent exposed 3 dots with a joker and you hold a 3 dot: a swap is on, and the "
        "closest hand has the fewest outs.",
        players=3,
        dealer=1,
        tiles=(
            *("3 bam",) * 3,
            *("3 crak",) * 3,
            "3 dot",
            "flower",
            "flower",
            "1 bam",
            "2 bam",
            "8 crak",
            "east wind",
            "9 dot",
        ),
        events=(
            discard(1, "9 dot"),
            discard(2, "3 dot"),
            {"type": "call", "player": 3, "tiles": ["3 dot", "3 dot", "joker"]},
            discard(3, "1 crak"),
            {"type": "draw", "tile": "7 dot"},
        ),
    ),
    Position(
        "bam-run",
        "A run with flowers is 3 away; like numbers shares its 3 bams but is 7 away.",
        players=2,
        dealer=2,
        tiles=(
            "1 bam",
            "1 bam",
            *("2 bam",) * 3,
            *("3 bam",) * 3,
            *("flower",) * 3,
            "3 crak",
            "3 dot",
        ),
    ),
    Position(
        "early",
        "Right after the deal: nothing is closer than 9 tiles and two concealed hands tie.",
        players=4,
        dealer=2,
        tiles=(
            "1 bam",
            "4 crak",
            "7 dot",
            "9 dot",
            "north wind",
            "red dragon",
            "white dragon",
            "flower",
            "2 crak",
            "5 bam",
            "8 bam",
            "east wind",
            "6 crak",
        ),
    ),
    Position(
        "locked-in",
        "You called four north winds, so only one hand is left to choose.",
        players=4,
        dealer=2,
        tiles=(
            *("5 bam",) * 3,
            "joker",
            "joker",
            *("north wind",) * 3,
            *("8 dot",) * 3,
            "2 crak",
            "4 crak",
        ),
        events=(
            discard(2, "north wind"),
            {"type": "call", "player": 1, "tiles": ["north wind"] * 4},
        ),
    ),
    Position(
        "sevens-drying-up",
        "Like numbers in 7s is closest, but opponents keep discarding the 7s it needs.",
        players=4,
        dealer=2,
        tiles=(
            *("7 bam",) * 3,
            "7 crak",
            "7 crak",
            "7 dot",
            "7 dot",
            "flower",
            "joker",
            "2 dot",
            "4 dot",
            "6 bam",
            "8 crak",
        ),
        events=(
            discard(2, "7 crak"),
            discard(3, "7 crak"),
            discard(4, "7 dot"),
            {"type": "draw", "tile": "1 bam"},
            discard(1, "1 bam"),
            discard(2, "7 dot"),
            discard(3, "flower"),
            discard(4, "flower"),
        ),
    ),
]
POSITIONS: dict[str, Position] = {position.name: position for position in _POSITIONS}


def sample_card() -> Card:
    return load_card(SAMPLE_CARD)


def candidates(card: Card, state: GameState) -> list[HandOdds]:
    """The hands the strategist chooses among in this position, closest first."""
    return advise(card, state, top=CANDIDATES).hands


def reads(card: Card, state: GameState) -> dict[int, Read]:
    """The engine's read of each opponent, as the app gives it to the strategist."""
    if state.phase != "play":
        return {}
    return {player: read_opponent(card, state, player) for player in range(2, state.players + 1)}
