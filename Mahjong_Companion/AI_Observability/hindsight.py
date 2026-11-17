# Showcase copy of packages/engine/src/mahjong_engine/hindsight.py from the private mahjong-companion repo. Here to be read, not run.

"""A game looked at again, move by move, for the game review. No AI here.

The game review (``mahjong_api.review``) asks Claude to coach the player once a game is over.
Claude is told only what this module works out by replaying the game, so every count it can
cite comes from the engine (ADR 1):

- each Charleston pass and each of the player's discards: what the engine would have done, what
  the player did, and how far the closest hand was afterwards either way;
- each discard the player could have called, and what they did with it;
- Mahjong the player had and didn't declare;
- each strategist pick: where it stood when it was asked, and how that hand fared from then on.

Move numbers count from 1, as the app's log and the play-by-play number them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mahjong_engine.calls import call_options
from mahjong_engine.card import Card
from mahjong_engine.charleston import suggest_pass
from mahjong_engine.game import ME, Game, GameState, advise, rank
from mahjong_engine.names import parse_tile
from mahjong_engine.odds import JOKER, HandOdds, closest_variants, rank_hands
from mahjong_engine.reads import read_opponent
from mahjong_engine.tiles import Tile

SEEN_LATER = 10  # tiles a pick needed that turned up later: the first few make the point
FAR = 99  # tiles away when no hand is possible


@dataclass(frozen=True)
class Pick:
    """A strategist pick to look at with hindsight."""

    ask: int  # its number in the game, from 1
    move: int  # moves played when it was asked
    hand_id: str
    followed: bool | None = None  # the player's "I'll go for it" / "Not this one" tap


def who(player: int | None) -> str:
    return "you" if player == ME else f"Player {player}"


def name(odds: HandOdds | None) -> str | None:
    return f"{odds.hand.label} ({odds.hand.id})" if odds else None


def facts(
    card: Card,
    game: Game,
    picks: Sequence[Pick] = (),
    opponent_cards: dict[int, Card] | None = None,
) -> dict[str, Any]:
    """Everything the review is told about one game, as plain data.

    ``card`` is the player's card; ``opponent_cards`` each opponent's, for reading which of the
    player's discards fed a hand an opponent seemed to be making (left out when not given).
    """
    return _Replay(card, game, list(picks), opponent_cards or {}).run()


class _Replay:
    def __init__(self, card: Card, game: Game, picks: list[Pick], opponent_cards: dict):
        self.card = card
        self.game = game
        self.picks = sorted(picks, key=lambda p: (p.move, p.ask))
        self.opponent_cards = opponent_cards
        self.state = GameState(game.players, game.dealer, game.messy)
        self.progress: list[dict[str, Any]] = []
        self.passes: list[dict[str, Any]] = []
        self.discards: list[dict[str, Any]] = []
        self.call_chances: list[dict[str, Any]] = []
        self.missed_mahjong: list[dict[str, Any]] = []
        self.asked: list[dict[str, Any]] = []
        self.watching: list[tuple[dict[str, Any], Counter[Tile]]] = []  # picks' missing tiles
        # Tiles away from the closest hand as last ranked, or None when it must be ranked again.
        # One more tile brings any hand at most one tile closer, and tiles seen elsewhere only
        # push hands away, so while this is 2 or more no single tile can make Mahjong. Only an
        # opponent's exposure can bring a hand closer between the player's moves (a joker to
        # swap for), so it resets this. It spares most of the ranking a long game would need.
        self.last_needed: int | None = None

    def run(self) -> dict[str, Any]:
        events = self.game.events
        self._picks_at(0)
        for move, event in enumerate(events, start=1):
            following = events[move] if move < len(events) else None
            self._play(move, event, following)
            self._picks_at(move)
        for pick in self.picks:
            if pick.move > len(events):
                self.asked.append(
                    {
                        "ask": pick.ask,
                        "move": pick.move,
                        "hand_id": pick.hand_id,
                        "note": "asked at a point in the game that was later undone",
                    }
                )
        self.asked.sort(key=lambda entry: entry["ask"])
        return self._summary()

    # One move.

    def _play(self, move: int, event: dict, following: dict | None) -> None:
        state, kind = self.state, event["type"]
        if kind == "pass":
            self._pass(move, event)
        elif kind == "discard" and int(event["player"]) == ME:
            self._my_discard(move, event, following)
        else:
            state.apply(event)
        if kind == "deal":
            self._note_progress(move, "deal")
        elif kind == "draw":
            self._after_draw(move, following)
        elif kind == "discard" and int(event["player"]) != ME:
            self._watch(event["tile"], f"discarded by {who(int(event['player']))}")
            self._call_chance(move, event, following)
        elif kind == "call":
            player = int(event["player"])
            called = state.exposures[player][-1].tile
            if player == ME:
                self._note_progress(move, f"called {called}")
            else:
                self._watch(called, f"exposed by {who(player)}")
                self.last_needed = None

    def _pass(self, move: int, event: dict) -> None:
        state = self.state
        direction = state.next_pass
        others = state.others_exposures
        suggested = suggest_pass(
            self.card, state.my_tiles, state.all_discards, others, size=len(event["out"])
        )
        before = self._closest()
        state.apply(event)
        passed = [str(t) for t in _tiles(event["out"])]
        after = self._closest()
        self.passes.append(
            {
                "move": move,
                "direction": direction,
                "you_passed": passed,
                "engine_would_pass": [str(t) for t in suggested],
                "same_tiles": _overlap(passed, [str(t) for t in suggested]),
                "you_received": [str(t) for t in _tiles(event["received"])],
                "tiles_away_before": before.needed if before else None,
                "tiles_away_after": after.needed if after else None,
                "closest_after": name(after),
            }
        )
        self._note_progress(move, "pass")

    def _my_discard(self, move: int, event: dict, following: dict | None) -> None:
        state = self.state
        tile = parse_tile(event["tile"])
        suggested = advise(self.card, state, ranked=[]).discard  # only the discard is wanted
        risky = self._risky(tile) if state.phase == "play" else []
        differs = suggested is not None and suggested != tile
        alternative = self._closest_without(suggested) if differs else None
        state.apply(event)
        after = self._closest()
        entry: dict[str, Any] = {
            "move": move,
            "you_discarded": str(tile),
            "engine_suggested": str(suggested) if suggested else None,
            "closest_after": name(after),
            "tiles_away_after": after.needed if after else None,
            "outs_after": after.outs if after else 0,
        }
        if differs:
            entry["tiles_away_if_suggested"] = alternative.needed if alternative else None
            entry["outs_if_suggested"] = alternative.outs if alternative else 0
        if risky:
            entry["an_opponent_may_need_it"] = risky
        if then := _what_happened_next(following, tile):
            entry["then"] = then
        self.discards.append(entry)
        self._note_progress(move, f"discarded {tile}")

    def _after_draw(self, move: int, following: dict | None) -> None:
        if self.last_needed is not None and self.last_needed > 1:
            return  # one tile can't complete a hand two or more tiles away
        closest = self._closest()
        declared = following is not None and following["type"] == "mahjong"
        if closest is not None and closest.needed == 0 and not declared:
            self.missed_mahjong.append(
                {"move": move, "how": "on your own draw", "hand": name(closest)}
            )

    def _call_chance(self, move: int, event: dict, following: dict | None) -> None:
        state = self.state
        took = _your_answer(following)
        held = state.concealed
        exposable = held[parse_tile(event["tile"])] + held[JOKER] >= 2  # a set needs 2 more
        near = self.last_needed is None or self.last_needed <= 1
        if took is None and not exposable and not near:
            return
        ranked = rank(self.card, state)
        self.last_needed = ranked[0].needed if ranked else FAR
        options = call_options(self.card, state, ranked)
        if not options and took is None:
            return
        best = options[0] if options else None
        if best is None:
            suggestion = "nothing: no call helps your closest hands"
        elif best.kind == "mahjong":
            suggestion = f"call it for Mahjong with {name_of(best.hand)}"
        else:
            suggestion = (
                f"call it and expose {', '.join(best.tiles)} toward {name_of(best.hand)} "
                f"({best.needed_before} -> {best.needed_after} tiles away)"
            )
        self.call_chances.append(
            {
                "move": move,
                "tile": str(parse_tile(event["tile"])),
                "discarded_by": who(int(event["player"])),
                "engine_suggested": suggestion,
                "you": took or "let it go",
            }
        )
        if best is not None and best.kind == "mahjong" and took != "declared Mahjong":
            self.missed_mahjong.append(
                {
                    "move": move,
                    "how": f"on {who(int(event['player']))}'s discard of {event['tile']}",
                    "hand": name_of(best.hand),
                }
            )

    # Strategist picks.

    def _picks_at(self, move: int) -> None:
        due = [p for p in self.picks if p.move == move]
        if not due or not self.state.my_tiles:
            return
        ranked = rank(self.card, self.state)
        ids = [odds.hand.id for odds in ranked]
        for pick in due:
            entry: dict[str, Any] = {
                "ask": pick.ask,
                "move": move,
                "hand_id": pick.hand_id,
                "you_followed": pick.followed,
                "engine_closest_then": name(ranked[0]) if ranked else None,
            }
            if pick.hand_id in ids:
                odds = ranked[ids.index(pick.hand_id)]
                missing = +(odds.variant.tiles - self.state.my_tiles)
                entry.update(
                    hand=name(odds),
                    engine_rank_then=ids.index(pick.hand_id) + 1,
                    tiles_away_then=odds.needed,
                    outs_then=odds.outs,
                    tiles_it_needed=sorted(str(t) for t in missing.elements()),
                    its_tiles_seen_later=[],
                )
                self.watching.append((entry, missing))
            else:
                entry["hand"] = f"{pick.hand_id} (no longer possible when asked)"
            self.asked.append(entry)

    def _watch(self, tile: Tile | str, how: str) -> None:
        tile = parse_tile(tile) if isinstance(tile, str) else tile
        for entry, missing in self.watching:
            seen = entry["its_tiles_seen_later"]
            if missing[tile] and len(seen) < SEEN_LATER:
                seen.append(f"{tile} {how}")

    # The end.

    def _summary(self) -> dict[str, Any]:
        state = self.state
        tiles = Counter(state.my_tiles)
        if state.winner == ME and state.last_discard and tiles.total() == 13:
            tiles[state.last_discard[1]] += 1  # Mahjong on a discard: the tile completes it
        final = self._closest(tiles) if tiles else None
        if state.phase != "over":
            result = "not over"
        elif state.winner is None:
            result = "wall game: nobody won"
        else:
            result = "you won" if state.winner == ME else f"{who(state.winner)} won"
        for entry in self.asked:
            if "tiles_away_then" not in entry:
                continue
            hand = next(h for h in self.card.hands if h.id == entry["hand_id"])
            at_end = closest_variants(
                hand, tiles, state.all_discards, state.others_exposures, state.exposures[ME]
            )
            entry["tiles_away_at_end"] = at_end[0].needed if at_end else None
            entry["still_possible_at_end"] = bool(at_end)
        return {
            "players": self.game.players,
            "you_were_east": self.game.dealer == ME,
            "charleston": not (self.game.messy or self.game.players == 2),
            "moves": len(self.game.events),
            "result": result,
            "your_closest_at_end": name(final),
            "tiles_away_at_end": final.needed if final else None,
            "your_exposures": [_exposure(e) for e in state.exposures[ME]],
            "others_exposures": {
                who(p): [_exposure(e) for e in es]
                for p, es in state.exposures.items()
                if p != ME and es
            },
            "your_progress": self.progress,
            "charleston_passes": self.passes,
            "your_discards": self.discards,
            "call_chances": self.call_chances,
            "missed_mahjong": self.missed_mahjong,
            "strategist_picks": self.asked,
        }

    # Helpers.

    def _closest(self, tiles: Counter[Tile] | None = None) -> HandOdds | None:
        state = self.state
        if tiles is None:
            ranked = rank(self.card, state) if state.my_tiles else []
        else:
            ranked = rank_hands(
                self.card, tiles, state.all_discards, state.others_exposures, state.exposures[ME]
            )
        return ranked[0] if ranked else None

    def _closest_without(self, tile: Tile) -> HandOdds | None:
        """The closest hand had the player let ``tile`` go instead."""
        state = self.state
        ranked = rank_hands(
            self.card,
            state.my_tiles - Counter({tile: 1}),
            [*state.all_discards, tile],
            state.others_exposures,
            state.exposures[ME],
        )
        return ranked[0] if ranked else None

    def _risky(self, tile: Tile) -> list[str]:
        """Opponents whose likely hands (on their own card) need ``tile``."""
        out = []
        for player, card in sorted(self.opponent_cards.items()):
            read = read_opponent(card, self.state, player)
            if not read.early and tile in read.careful:
                out.append(who(player))
        return out

    def _note_progress(self, move: int, after: str) -> None:
        closest = self._closest()
        self.last_needed = closest.needed if closest else FAR
        if closest is not None:
            self.progress.append(
                {
                    "move": move,
                    "after": after,
                    "closest": name(closest),
                    "tiles_away": closest.needed,
                }
            )


def name_of(hand: Any) -> str:
    return f"{hand.label} ({hand.id})"


def _tiles(names: Iterable[str]) -> list[Tile]:
    return [parse_tile(n) for n in names]


def _overlap(a: list[str], b: list[str]) -> int:
    return (Counter(a) & Counter(b)).total()


def _exposure(e: Any) -> str:
    jokers = f" with {e.jokers} joker{'s' if e.jokers > 1 else ''}" if e.jokers else ""
    return f"{e.size}x {e.tile}{jokers}"


def _your_answer(following: dict | None) -> str | None:
    """What the player did with the discard just made, judging by the next move."""
    if following is None:
        return None
    if following["type"] == "call" and int(following["player"]) == ME:
        return f"called it and exposed {', '.join(following['tiles'])}"
    if following["type"] == "mahjong" and int(following["player"]) == ME:
        return "declared Mahjong"
    return None


def _what_happened_next(following: dict | None, tile: Tile) -> str | None:
    if following is None:
        return None
    player = following.get("player")
    if following["type"] == "call" and player is not None and int(player) != ME:
        return f"{who(int(player))} called it"
    if following["type"] == "mahjong" and player is not None and int(player) != ME:
        return f"{who(int(player))} declared Mahjong right after (on it, or on their own draw)"
    return None
