# Showcase copy of services/api/src/mahjong_api/jev.py from the private mahjong-companion repo. Here to be read, not run.

"""Jev decides the table talk the grammar parser can't place (ADR 4, Phase 6).

Two typed questions about one transcript line: is it a player announcing a discard (a yes/no
"noul" probability), and which of the 36 tiles does it name (a choice with a probability per
tile). Jev writes no text and costs $0.042 per million input tokens, about 400 tokens a line.

The key is ``TYPESAFE_API_KEY`` (in Azure, from Key Vault); locally ``TYPESAFE_JEV_KEY`` also
works. Jev calls are traced in Langfuse by hand, since TypeSafe has no integration.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from mahjong_engine.tiles import FULL_SET, TileKind

MODEL = "jev-1.13.0"  # pinned: thresholds below were set against this version
INPUT_PRICE_PER_TOKEN = 0.042 / 1_000_000  # output tokens are free
# Calibrated on 12 lines (Oct 30, jev-1.13.0): announcements scored 0.93, 0.75, 0.75, 0.60,
# 0.48, 0.17; table chatter 0.37, 0.23, 0.07, 0.02, 0.02, 0.02. TypeSafe's suggested 0.7 would
# miss half the announcements; 0.4 keeps 5 of 6 and drops all chatter. Every proposal is
# confirmed by the player, so a stray card costs one tap. Revisit with discard_parse_correct.
GAME_TALK = 0.4

SUIT_NAMES = {TileKind.BAM: "bamboo (bam)", TileKind.CRAK: "character (crak)", TileKind.DOT: "dot"}


def _tile_criteria() -> dict[str, str]:
    criteria = {}
    for tile in sorted(FULL_SET):
        name = str(tile)
        if tile.kind in SUIT_NAMES:
            criteria[name] = f"the {tile.value} of {SUIT_NAMES[tile.kind]}"
        elif name == "white dragon":
            criteria[name] = "the white dragon, also called soap or zero"
        elif tile.kind == TileKind.DRAGON:
            criteria[name] = f"the {tile.value} dragon"
        elif tile.kind == TileKind.WIND:
            criteria[name] = f"the {tile.value} wind, often just '{tile.value}'"
        else:
            criteria[name] = f"a {name}"
    return criteria


TILE_CRITERIA = _tile_criteria()


@dataclass(frozen=True)
class Verdict:
    game_talk: float  # probability the line announces a discard
    tile: str  # the most likely tile name
    tile_probability: float
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost(self) -> float:
        return self.input_tokens * INPUT_PRICE_PER_TOKEN


def get_jev() -> Iterator[Any | None]:
    """A Jev client for one request, or None when no key is configured."""
    key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_JEV_KEY")
    if not key:
        yield None
        return
    from typesafe_sdk import TypeSafeClient

    with TypeSafeClient(api_key=key, model=MODEL, timeout=10) as client:
        yield client


def judge(client: Any, text: str, turn: str) -> Verdict:
    """Ask Jev whether a line is a discard announcement, and which tile it names."""
    from typesafe_sdk import Choice, Noul

    response = client.system_one(
        state={
            "line": text,
            "setting": "One line of speech-to-text from an American Mahjong table. "
            f"It is {turn}'s turn. Number words are often misheard: 'for' = four, "
            "'free' or 'tree' = three, 'ate' = eight, 'won' = one, 'to' = two.",
        },
        questions={
            "game_talk": Noul(
                instructions="Is this line a player naming the tile they are discarding now?",
                criteria={
                    "true": "Announces a discard, e.g. 'three bam', 'I'll let go of the soap'.",
                    "false": "Conversation, a question, or a tile someone wants or holds.",
                },
            ),
            "tile": Choice(
                instructions="Which tile does the line name? Pick the closest even if unsure.",
                criteria=TILE_CRITERIA,
            ),
        },
    )
    game_talk = _field(response.answers["game_talk"], "noul")
    tile_answer = response.answers["tile"]
    tile = _field(tile_answer, "choice")
    probabilities = _field(tile_answer, "probabilities") or {}
    usage = response.usage
    return Verdict(
        game_talk=float(game_talk),
        tile=tile,
        tile_probability=float(probabilities.get(tile, 0.0)),
        model=response.model,
        input_tokens=_field(usage, "input_tokens") or 0,
        output_tokens=_field(usage, "output_tokens") or 0,
    )


def _field(value: Any, name: str) -> Any:
    """Answers may arrive as dicts or objects, depending on the SDK version."""
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)
