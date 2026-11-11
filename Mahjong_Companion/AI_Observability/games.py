# Showcase copy of services/api/src/mahjong_api/games.py from the private mahjong-companion repo. Here to be read, not run.

"""Game routes: start a game, add events, undo or fix them, and read the state with advice.

Each game is one JSON file in the store (``games/<id>.json``) holding its event list; the state is
replayed from the events on every read, so an undo or a fix is just an edited list.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from mahjong_engine.calls import CallOption, call_options
from mahjong_engine.card import Card, card_from_dict
from mahjong_engine.game import ME, Game, GameError, GameState, advise
from mahjong_engine.odds import Exposure, HandOdds
from mahjong_engine.reads import Read, read_opponent
from mahjong_engine.tiles import Tile
from pydantic import BaseModel, Field

from mahjong_api import jev, listen, playbyplay, strategist, vision
from mahjong_api.store import Store, store_from_env

router = APIRouter(prefix="/api/games")
cards_router = APIRouter(prefix="/api/cards")
_cards: dict[str, Card] = {}


def get_store() -> Store:
    return store_from_env()


StoreDep = Annotated[Store, Depends(get_store)]


def get_claude() -> Any | None:
    """An Anthropic client, or None when no API key is configured."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    return anthropic.Anthropic()


ClaudeDep = Annotated[Any, Depends(get_claude)]


def default_card() -> str:
    """The year of the card players use unless the game says otherwise (``CARD_NAME``)."""
    return Path(os.environ.get("CARD_NAME", "cards/2024.toml")).stem


def card_for(game: Game, player: int) -> str:
    return game.cards.get(player) or default_card()


def available_cards(store: Store) -> list[str]:
    """Years with a card file in the store, e.g. ["2023", "2024"]."""
    return sorted(Path(name).stem for name in store.list("cards") if name.endswith(".toml"))


def load_card(store: Store, year: str | None = None) -> Card | None:
    year = year or default_card()
    if not year.isalnum():
        return None
    name = f"cards/{year}.toml"
    if name not in _cards:
        text = store.get(name)
        if text is None:
            return None
        _cards[name] = card_from_dict(tomllib.loads(text))
    return _cards[name]


@cards_router.get("")
def list_cards(store: StoreDep) -> dict[str, Any]:
    """The cards players can choose from, and the default."""
    return {"cards": available_cards(store), "default": default_card()}


class NewGame(BaseModel):
    players: int = Field(4, ge=2, le=4)
    dealer: int = Field(1, ge=1, le=4)
    messy: bool = False
    cards: dict[int, str] = Field(default_factory=dict)  # player -> card year


@router.post("")
def create_game(body: NewGame, store: StoreDep) -> dict[str, Any]:
    known = available_cards(store)
    for player, year in body.cards.items():
        if not 1 <= player <= body.players:
            raise HTTPException(400, f"No player {player} in a {body.players}-player game")
        if year not in known:
            raise HTTPException(400, f"No {year} card; known cards: {', '.join(known) or 'none'}")
    game = Game(body.players, body.dealer, body.messy, cards=dict(body.cards))
    _check(game)
    game_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
    _save(store, game_id, game)
    return _view(store, game_id, game)


@router.get("")
def list_games(store: StoreDep) -> list[dict[str, Any]]:
    games = []
    for name in reversed(store.list("games")):
        game_id = name.removeprefix("games/").removesuffix(".json")
        game = _load(store, game_id)
        state = game.state()
        games.append({"id": game_id, "players": game.players, "phase": state.phase})
    return games


@router.get("/{game_id}")
def get_game(game_id: str, store: StoreDep) -> dict[str, Any]:
    return _view(store, game_id, _load(store, game_id))


@router.post("/{game_id}/events")
def add_event(game_id: str, event: dict[str, Any], store: StoreDep) -> dict[str, Any]:
    game = _load(store, game_id)
    try:
        game.add(event)
    except (GameError, ValueError, KeyError) as e:
        raise HTTPException(400, str(e)) from e
    _save(store, game_id, game)
    view = _view(store, game_id, game)
    playbyplay.record_move(game_id, len(game.events) - 1, event, view)
    return view


@router.put("/{game_id}/events/{index}")
def fix_event(game_id: str, index: int, event: dict[str, Any], store: StoreDep) -> dict[str, Any]:
    """Replace one event; every later event must still make sense."""
    game = _load(store, game_id)
    if not 0 <= index < len(game.events):
        raise HTTPException(404, f"No event {index}")
    before = game.events[index]
    game.events[index] = event
    _check(game)
    _save(store, game_id, game)
    view = _view(store, game_id, game)
    playbyplay.record_fix(game_id, index, before, event, view)
    return view


@router.delete("/{game_id}/events/last")
def undo(game_id: str, store: StoreDep) -> dict[str, Any]:
    game = _load(store, game_id)
    if not game.events:
        raise HTTPException(400, "Nothing to undo")
    removed = game.events.pop()
    _save(store, game_id, game)
    view = _view(store, game_id, game)
    playbyplay.record_undo(game_id, removed, view)
    return view


@router.post("/{game_id}/strategist")
def ask_strategist(game_id: str, store: StoreDep, claude: ClaudeDep) -> dict[str, Any]:
    """Claude's pick among the engine's closest hands, with a one-line reason."""
    if claude is None:
        raise HTTPException(503, "The strategist isn't set up: no ANTHROPIC_API_KEY")
    game = _load(store, game_id)
    card = load_card(store, card_for(game, ME))
    if card is None:
        raise HTTPException(503, "No card loaded")
    state = game.state()
    if not state.my_tiles or state.phase == "over":
        raise HTTPException(400, "Ask once your tiles are in and the game is on")
    reads = _reads(store, game, state)
    import anthropic

    try:
        pick = strategist.ask(claude, card, state, game_id, reads)
    except strategist.StrategistError as e:
        raise HTTPException(502, str(e)) from e
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {e}") from e
    return {
        "hand_id": pick.hand_id,
        "label": pick.label,
        "reason": pick.reason,
        "trace_id": pick.trace_id,
    }


class Feedback(BaseModel):
    trace_id: str
    hand_id: str
    followed: bool


@router.post("/{game_id}/strategist/feedback")
def strategist_feedback(game_id: str, body: Feedback) -> dict[str, bool]:
    """Record whether the player went for the strategist's pick, as a Langfuse score."""
    langfuse = strategist.langfuse_client()
    langfuse.create_score(
        name="recommendation_followed",
        value=1.0 if body.followed else 0.0,
        data_type="BOOLEAN",
        trace_id=body.trace_id,
        metadata={"game": game_id, "hand_id": body.hand_id},
    )
    langfuse.flush()
    return {"recorded": True}


class Photo(BaseModel):
    image: str  # base64, already shrunk by the app
    media_type: str = "image/jpeg"


@router.post("/{game_id}/photo")
def read_photo(game_id: str, body: Photo, store: StoreDep, claude: ClaudeDep) -> dict[str, Any]:
    """Read the tiles in a photo of your hand, for you to check before dealing them in."""
    if claude is None:
        raise HTTPException(503, "Photo reading isn't set up: no ANTHROPIC_API_KEY")
    _load(store, game_id)  # 404 for an unknown game
    import anthropic

    try:
        reading = vision.read_hand(claude, body.image, body.media_type, game_id)
    except vision.VisionError as e:
        raise HTTPException(400, str(e)) from e
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {e}") from e
    # Keep the photo privately: with the player's corrected tiles it becomes training data
    # for a tile detector (Phase 5). The label is written when the player confirms the tiles.
    photo_id = f"{game_id}-{secrets.token_hex(4)}"
    extension = body.media_type.split("/")[1].replace("jpeg", "jpg")
    store.put_bytes(
        f"hand-photos/{photo_id}.{extension}", base64.b64decode(body.image), body.media_type
    )
    return {
        "tiles": reading.tiles,
        "uncertain": reading.uncertain,
        "trace_id": reading.trace_id,
        "photo_id": photo_id,
    }


class PhotoFeedback(BaseModel):
    trace_id: str | None = None
    photo_id: str | None = None
    read: list[str]
    final: list[str]


@router.post("/{game_id}/photo/feedback")
def photo_feedback(game_id: str, body: PhotoFeedback, store: StoreDep) -> dict[str, float]:
    """Score a reading by how many tiles you fixed, and label the saved photo with the truth."""
    correct = vision.score_reading(body.trace_id, body.read, body.final) if body.trace_id else 0.0
    if (
        body.photo_id
        and body.photo_id.startswith(f"{game_id}-")
        and body.photo_id.replace("-", "").isalnum()
    ):
        label = {"game": game_id, "tiles": body.final, "read": body.read, "trace_id": body.trace_id}
        store.put(f"hand-photos/{body.photo_id}.json", json.dumps(label, indent=1))
    return {"correct": correct}


class Line(BaseModel):
    text: str = Field(max_length=500)


@router.post("/{game_id}/heard")
def heard(
    game_id: str, body: Line, store: StoreDep, jev_client: Annotated[Any, Depends(jev.get_jev)]
) -> dict[str, Any]:
    """Propose the event a line of table talk most likely means, for one-tap confirm or fix."""
    return listen.hear(_load(store, game_id).state(), body.text, game_id, jev_client)


class HeardFeedback(BaseModel):
    trace_id: str
    outcome: Literal["confirmed", "auto", "fixed", "ignored"]


@router.post("/{game_id}/heard/feedback")
def heard_feedback(game_id: str, body: HeardFeedback) -> dict[str, float]:
    """Score a proposal by the player's answer (Langfuse ``discard_parse_correct``)."""
    return {"correct": listen.score(body.trace_id, body.outcome)}


def _check(game: Game) -> None:
    try:
        game.state()
    except (GameError, ValueError, KeyError) as e:
        raise HTTPException(400, str(e)) from e


def _load(store: Store, game_id: str) -> Game:
    if not game_id.replace("-", "").isalnum():
        raise HTTPException(404, "No such game")
    text = store.get(f"games/{game_id}.json")
    if text is None:
        raise HTTPException(404, "No such game")
    return Game.from_dict(json.loads(text))


def _save(store: Store, game_id: str, game: Game) -> None:
    store.put(f"games/{game_id}.json", json.dumps(game.to_dict(), indent=1))


def _names(tiles: Counter[Tile] | list[Tile]) -> list[str]:
    items = sorted(tiles.elements()) if isinstance(tiles, Counter) else tiles
    return [str(t) for t in items]


def _exposure(e: Exposure) -> dict[str, Any]:
    return {"tile": str(e.tile), "size": e.size, "jokers": e.jokers}


def _hand(odds: HandOdds) -> dict[str, Any]:
    return {
        "id": odds.hand.id,
        "label": odds.hand.label,
        "concealed": odds.hand.concealed,
        "needed": odds.needed,
        "outs": odds.outs,
        "swaps": odds.swaps,
        "target": [{"tile": str(t), "count": n} for t, n in odds.variant.sets],
    }


def _view(store: Store, game_id: str, game: Game) -> dict[str, Any]:
    state: GameState = game.state()
    my_card = card_for(game, ME)
    card = load_card(store, my_card)
    advice = None
    if card is not None and state.my_tiles:
        a = advise(card, state)
        advice = {
            "hands": [_hand(h) for h in a.hands],
            "pass": _names(a.pass_tiles),
            "discard": str(a.discard) if a.discard else None,
            "calls": [_call(c) for c in call_options(card, state)],
        }
    last = state.last_discard
    return {
        "id": game_id,
        "players": game.players,
        "dealer": game.dealer,
        "messy": game.messy,
        "phase": state.phase,
        "turn": state.turn,
        "next_pass": state.next_pass,
        "winner": state.winner,
        "my_tiles": _names(state.my_tiles),
        "concealed": _names(state.concealed),
        "exposures": {p: [_exposure(e) for e in es] for p, es in state.exposures.items()},
        "discards": {p: _names(ts) for p, ts in state.discards.items()},
        "last_discard": {"player": last[0], "tile": str(last[1])} if last else None,
        "events": game.events,
        "card": my_card if card else None,
        "cards": {p: card_for(game, p) for p in range(1, game.players + 1)},
        "advice": advice,
        "reads": {p: _read(r) for p, r in _reads(store, game, state).items()},
    }


def _reads(store: Store, game: Game, state: GameState) -> dict[int, Read]:
    """Each opponent's likely hands, on that opponent's own card."""
    if state.phase != "play":
        return {}
    reads = {}
    for player in range(2, game.players + 1):
        card = load_card(store, card_for(game, player))
        if card is not None:
            reads[player] = read_opponent(card, state, player)
    return reads


def _call(option: CallOption) -> dict[str, Any]:
    return {
        "kind": option.kind,
        "hand_id": option.hand.id,
        "label": option.hand.label,
        "tiles": option.tiles,
        "needed_before": option.needed_before,
        "needed_after": option.needed_after,
        "rules_out_concealed": option.rules_out_concealed,
    }


def _read(read: Read) -> dict[str, Any]:
    return {
        "early": read.early,
        "likely": [
            {"id": x.hand.id, "label": x.hand.label, "target": str(x.variant)} for x in read.likely
        ],
        "careful": [str(t) for t in read.careful],
        "discarded_suits": read.discarded_suits,
    }
