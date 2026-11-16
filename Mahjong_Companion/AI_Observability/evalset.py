# Showcase copy of services/api/src/mahjong_api/evalset.py from the private mahjong-companion repo. Here to be read, not run.

"""Fixed, made-up test material for measuring the AI parts over time (ADR 7).

Everything here is played on the made-up sample card (``packages/engine/tests``), never a real
NMJL card, so it can live in git and in Langfuse datasets.

- ``POSITIONS``: game positions the strategist is asked about again and again. The inputs never
  change, so a change in its answers, tokens or latency is the model or the prompt, not the games
  (``scripts/strategist_eval.py``).
- ``JUDGE_CASES``: strategist answers to those positions, good and deliberately broken, each
  labelled by hand with the verdict a careful person would give. They measure whether a local
  model is fit to judge, and whether it still is after an update
  (``scripts/judge.py calibrate``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class JudgeCase:
    """One strategist answer with the verdicts a careful person gives it.

    ``expect`` maps a check to True (should pass) or False (should fail). A check left out is a
    judgment call that isn't counted. ``relies_on`` pins the engine's numbers the labels were
    written against (hand -> tiles needed, outs), so a change in the engine can't quietly make
    a label wrong.
    """

    name: str
    position: str
    hand_id: str
    reason: str
    expect: dict[str, bool]
    tools: tuple[str, ...] = ()  # tool results the strategist is shown to have looked up
    relies_on: dict[str, tuple[int, int]] = field(default_factory=dict)
    why: str = ""


PASS_ALL = {"grounded": True, "matches_pick": True, "sound_choice": True, "clear": True}

JUDGE_CASES = [
    # --- triple-threes: Like Numbers top line needs 4 (12 outs); every other hand needs 9+.
    JudgeCase(
        "threes-good",
        "triple-threes",
        "1999-LIKE-1",
        "Go for like numbers in 3s because you need only 4 tiles and already hold three 3 bams, "
        "three 3 craks and two 3 dots.",
        PASS_ALL,
        relies_on={"1999-LIKE-1": (4, 12)},
        why="Every fact is on the table and the hand is the closest.",
    ),
    JudgeCase(
        "threes-wrong-numbers",
        "triple-threes",
        "1999-LIKE-1",
        "Go for like numbers in 3s because you need only 2 tiles and have 20 outs with your "
        "3 bams and 3 craks.",
        {**PASS_ALL, "grounded": False},
        relies_on={"1999-LIKE-1": (4, 12)},
        why="The engine says 4 tiles and 12 outs, not 2 and 20.",
    ),
    JudgeCase(
        "threes-invented-jokers",
        "triple-threes",
        "1999-LIKE-1",
        "Go for like numbers in 3s because your pair of jokers and three 3 bams put you 4 tiles "
        "away.",
        {**PASS_ALL, "grounded": False},
        relies_on={"1999-LIKE-1": (4, 12)},
        why="The player holds no jokers.",
    ),
    JudgeCase(
        "threes-far-hand-no-case",
        "triple-threes",
        "1999-RUN-2",
        "Go for the run of 3 bams, north winds and 6 dots because you already hold three 3 bams "
        "and a 6 dot.",
        {**PASS_ALL, "sound_choice": False},
        relies_on={"1999-LIKE-1": (4, 12), "1999-RUN-2": (10, 16)},
        why="True facts, but it needs 10 tiles against 4 and names no advantage over the "
        "closer hand.",
    ),
    JudgeCase(
        "threes-vague",
        "triple-threes",
        "1999-LIKE-1",
        "This one is the best choice for you right now.",
        {**PASS_ALL, "clear": False},
        relies_on={"1999-LIKE-1": (4, 12)},
        why="Nothing false, but no hand, tile or number: the player learns nothing.",
    ),
    JudgeCase(
        "threes-describes-another-hand",
        "triple-threes",
        "1999-LIKE-1",
        "Go for the evens hand at the bottom of the runs because it has 44 outs.",
        {"grounded": False, "matches_pick": False, "clear": True},
        relies_on={"1999-LIKE-1": (4, 12), "1999-RUN-3": (11, 44)},
        why="The pick is like numbers; the reason argues for evens with that hand's outs.",
    ),
    JudgeCase(
        "threes-names-the-wrong-line",
        "triple-threes",
        "1999-LIKE-2",
        "Go for like numbers on the top line: four each of the 3 bams, 3 craks and 3 dots with "
        "your two flowers.",
        {"grounded": False, "matches_pick": False, "clear": True},
        relies_on={"1999-LIKE-2": (9, 32)},
        why="The pick is the bottom line (winds and dragons); the reason describes the top line.",
    ),
    # --- winds-and-fives: Like Numbers bottom line (concealed) needs 4 (16 outs).
    JudgeCase(
        "winds-good",
        "winds-and-fives",
        "1999-LIKE-2",
        "Go for the winds and dragons hand because you hold all four winds, two 5 bams and two "
        "green dragons, and need only 4 more tiles.",
        PASS_ALL,
        relies_on={"1999-LIKE-2": (4, 16)},
        why="Every fact checks out and it is the closest hand.",
    ),
    JudgeCase(
        "winds-false-opponent-claim",
        "winds-and-fives",
        "1999-LIKE-2",
        "Go for the winds and dragons hand because Player 2 has thrown three green dragons, so "
        "nobody else wants them.",
        {**PASS_ALL, "grounded": False},
        tools=("opponent_reads",),
        relies_on={"1999-LIKE-2": (4, 16)},
        why="The opponent read shows Player 2 has discarded nothing.",
    ),
    JudgeCase(
        "winds-far-hand-false-case",
        "winds-and-fives",
        "1999-RUN-3",
        "Go for the evens hand at the bottom because it needs only 3 tiles, fewer than any other "
        "hand.",
        {**PASS_ALL, "grounded": False, "sound_choice": False},
        relies_on={"1999-LIKE-2": (4, 16), "1999-RUN-3": (11, 44)},
        why="It needs 11 tiles, so the one advantage it claims is false.",
    ),
    JudgeCase(
        "winds-far-hand-true-case",
        "winds-and-fives",
        "1999-RUN-1",
        "Go for the dot run with green dragons because it has 20 outs against the closest "
        "hand's 16, and it is not concealed.",
        PASS_ALL,
        relies_on={"1999-LIKE-2": (4, 16), "1999-RUN-1": (10, 20)},
        why="Farther away, but it names real advantages the engine's numbers confirm. Whether "
        "they are worth it is the player's call, not this check's.",
    ),
    # --- evens: Runs bottom line (concealed singles) needs 3 (10 outs).
    JudgeCase(
        "evens-good",
        "evens",
        "1999-RUN-3",
        "Go for the evens hand because you need only 3 tiles: the 8 crak, the 6 dot and the 8 dot.",
        PASS_ALL,
        relies_on={"1999-RUN-3": (3, 10)},
        why="Those are exactly the three target tiles the player lacks.",
    ),
    JudgeCase(
        "evens-joker-for-a-single",
        "evens",
        "1999-RUN-3",
        "Go for the evens hand because you need 3 tiles and your joker can stand in for the 8 dot.",
        {**PASS_ALL, "grounded": False},
        relies_on={"1999-RUN-3": (3, 10)},
        why="A joker may only fill a set of three or more, never a single.",
    ),
    JudgeCase(
        "evens-rambling",
        "evens",
        "1999-RUN-3",
        "Well, there are several options here. The evens hand looks good. But the like numbers "
        "could work too. I would probably lean toward evens, maybe.",
        {**PASS_ALL, "clear": False},
        relies_on={"1999-RUN-3": (3, 10)},
        why="Four hedging sentences and no tiles or numbers.",
    ),
    # --- dot-run: Runs top line needs 2 (8 outs); both other red dragons are discarded.
    JudgeCase(
        "dots-good",
        "dot-run",
        "1999-RUN-1",
        "Go for the dot run with red dragons because you are only 2 tiles away, holding three "
        "each of the 1, 2 and 3 dots.",
        PASS_ALL,
        relies_on={"1999-RUN-1": (2, 8)},
        why="Every fact checks out and it is the closest hand.",
    ),
    JudgeCase(
        "dots-far-hand-true-case",
        "dot-run",
        "1999-LIKE-2",
        "Go for the winds and dragons hand because both other red dragons are already "
        "discarded, which starves the dot run.",
        PASS_ALL,
        tools=("opponent_reads",),
        relies_on={"1999-RUN-1": (2, 8), "1999-LIKE-2": (8, 31)},
        why="Players 2 and 3 did each discard a red dragon: a real problem for the closer hand.",
    ),
    JudgeCase(
        "dots-describes-another-hand",
        "dot-run",
        "1999-RUN-2",
        "Go for the dot run with red dragons because you are only 2 tiles away.",
        {"grounded": False, "matches_pick": False, "clear": True},
        relies_on={"1999-RUN-1": (2, 8), "1999-RUN-2": (9, 15)},
        why="The pick is the quints-and-winds line, 9 away; the reason describes the dot run.",
    ),
    JudgeCase(
        "dots-far-hand-for-fun",
        "dot-run",
        "1999-RUN-3",
        "Go for the evens hand because it is the most fun to play.",
        {**PASS_ALL, "sound_choice": False, "clear": False},
        relies_on={"1999-RUN-1": (2, 8), "1999-RUN-3": (13, 51)},
        why="13 tiles away against 2, and no tile, number or advantage given.",
    ),
]


def judge_material(case: JudgeCase) -> dict[str, Any]:
    """What the judge is shown for a labelled case, built by the engine from its position."""
    from mahjong_api import strategist

    position = POSITIONS[case.position]
    card, state = sample_card(), position.state()
    hands = candidates(card, state)
    by_id = {odds.hand.id: odds for odds in hands}
    tools = []
    for name in case.tools:
        if name == "opponent_reads":
            tools.append(
                {"name": name, "input": {}, "output": strategist._opponent_reads(state, {})}
            )
        elif name == "hand_details":
            details = strategist._hand_details(card, state, by_id[case.hand_id])
            tools.append({"name": name, "input": {"hand_id": case.hand_id}, "output": details})
    return {
        "situation": strategist._situation(state, hands),
        "candidates": [strategist._odds(odds) for odds in hands],
        "tools": tools,
        "hand_id": case.hand_id,
        "label": by_id[case.hand_id].hand.label,
        "reason": case.reason,
        "engine_rank": list(by_id).index(case.hand_id) + 1,
    }
