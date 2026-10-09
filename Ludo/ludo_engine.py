"""Core Ludo rules engine.

The engine stores each piece by its player-relative distance:
- -1 means the piece is still in the starting circle.
- 0..51 means the piece is on the shared 52-space track.
- 52..56 means the piece is in that player's private home column.
- 57 means the piece has reached the home triangle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BASE = -1
PATH_START = 0
PATH_END = 51
HOME_COLUMN_START = 52
HOME_COLUMN_END = 56
FINISHED = 57
PIECES_PER_PLAYER = 4

RED = "RED"
YELLOW = "YELLOW"
BLUE = "BLUE"
GREEN = "GREEN"

PLAYER_ORDER: Tuple[str, ...] = (RED, YELLOW, BLUE, GREEN)


@dataclass(frozen=True)
class PlayerSpec:
    color: str
    display_name: str
    start_index: int
    short_name: str


PLAYER_SPECS: Dict[str, PlayerSpec] = {
    RED: PlayerSpec(RED, "Red", 0, "R"),
    YELLOW: PlayerSpec(YELLOW, "Yellow", 13, "Y"),
    BLUE: PlayerSpec(BLUE, "Blue", 26, "B"),
    GREEN: PlayerSpec(GREEN, "Green", 39, "G"),
}

SAFE_ABSOLUTE_SQUARES = frozenset(spec.start_index for spec in PLAYER_SPECS.values())


@dataclass(frozen=True)
class MoveOption:
    player: str
    piece_index: int
    source: int
    target: int
    target_absolute: Optional[int]
    bumps: Tuple[Tuple[str, int], ...] = ()

    @property
    def is_from_base(self) -> bool:
        return self.source == BASE

    @property
    def reaches_home_triangle(self) -> bool:
        return self.target == FINISHED

    @property
    def enters_home_column(self) -> bool:
        return self.source <= PATH_END and self.target >= HOME_COLUMN_START

    @property
    def lands_on_safe_square(self) -> bool:
        return self.target_absolute in SAFE_ABSOLUTE_SQUARES


@dataclass(frozen=True)
class RollOutcome:
    player: str
    die: int
    legal_moves: Tuple[MoveOption, ...] = ()
    passed: bool = False
    turn_lost: bool = False


@dataclass(frozen=True)
class MoveResult:
    player: str
    piece_index: int
    die: int
    source: int
    target: int
    bumps: Tuple[Tuple[str, int], ...] = ()
    extra_turn: bool = False
    winner: Optional[str] = None


@dataclass
class PlayerStats:
    turns_started: int = 0
    rolls: int = 0
    moves: int = 0
    passes: int = 0
    sixes: int = 0
    extra_turns: int = 0
    three_six_losses: int = 0
    bumps_given: int = 0
    times_bumped: int = 0
    base_exits: int = 0
    home_entries: int = 0
    safe_landings: int = 0
    pieces_finished: int = 0
    total_pips_moved: int = 0


@dataclass
class GameState:
    pieces: Dict[str, List[int]]
    active_players: Tuple[str, ...] = PLAYER_ORDER
    start_indices: Dict[str, int] = field(default_factory=dict)
    current_player: str = RED
    pending_roll: Optional[int] = None
    consecutive_sixes: int = 0
    turn_count: int = 1
    winner: Optional[str] = None
    stats: Dict[str, PlayerStats] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)


def create_new_game(active_players: Optional[Sequence[str]] = None) -> GameState:
    normalized_players = normalize_active_players(active_players)
    first_player = RED if RED in normalized_players else normalized_players[0]
    state = GameState(
        pieces={player: [BASE] * PIECES_PER_PLAYER for player in PLAYER_ORDER},
        active_players=normalized_players,
        start_indices=create_start_indices(normalized_players),
        current_player=first_player,
        stats={player: PlayerStats() for player in PLAYER_ORDER},
        log=[f"New game started. {player_label(first_player)} moves first."],
    )
    state.stats[first_player].turns_started = 1
    return state


def clone_state(state: GameState) -> GameState:
    return GameState(
        pieces={player: list(piece_positions) for player, piece_positions in state.pieces.items()},
        active_players=tuple(state.active_players),
        start_indices=dict(state.start_indices),
        current_player=state.current_player,
        pending_roll=state.pending_roll,
        consecutive_sixes=state.consecutive_sixes,
        turn_count=state.turn_count,
        winner=state.winner,
        stats={
            player: PlayerStats(**vars(stats))
            for player, stats in state.stats.items()
        },
        log=list(state.log),
    )


def normalize_active_players(active_players: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if active_players is None:
        return PLAYER_ORDER

    unique_players = tuple(
        player
        for player in PLAYER_ORDER
        if player in set(active_players)
    )
    if not unique_players:
        raise ValueError("At least one player must be active.")
    return unique_players


def create_start_indices(active_players: Sequence[str]) -> Dict[str, int]:
    start_indices = {
        player: spec.start_index
        for player, spec in PLAYER_SPECS.items()
    }
    if len(active_players) == 2:
        first_player, second_player = active_players
        start_indices[second_player] = (start_indices[first_player] + 26) % 52
    return start_indices


def player_label(player: str) -> str:
    return PLAYER_SPECS[player].display_name


def piece_label(player: str, piece_index: int) -> str:
    return f"{PLAYER_SPECS[player].short_name}{piece_index + 1}"


def next_player(player: str, active_players: Sequence[str] = PLAYER_ORDER) -> str:
    if not active_players:
        raise ValueError("At least one player must be active.")
    if player not in active_players:
        return active_players[0]
    index = tuple(active_players).index(player)
    return active_players[(index + 1) % len(active_players)]


def is_on_shared_track(relative_position: int) -> bool:
    return PATH_START <= relative_position <= PATH_END


def is_in_home_column(relative_position: int) -> bool:
    return HOME_COLUMN_START <= relative_position <= HOME_COLUMN_END


def absolute_position(player: str, relative_position: int) -> Optional[int]:
    if not is_on_shared_track(relative_position):
        return None
    return (PLAYER_SPECS[player].start_index + relative_position) % 52


def state_absolute_position(state: GameState, player: str, relative_position: int) -> Optional[int]:
    if not is_on_shared_track(relative_position):
        return None
    return (state.start_indices.get(player, PLAYER_SPECS[player].start_index) + relative_position) % 52


def relative_index_for_absolute(player: str, absolute_index: int) -> int:
    return (absolute_index - PLAYER_SPECS[player].start_index) % 52


def state_relative_index_for_absolute(state: GameState, player: str, absolute_index: int) -> int:
    return (absolute_index - state.start_indices.get(player, PLAYER_SPECS[player].start_index)) % 52


def move_target(relative_position: int, die: int) -> Optional[int]:
    if die < 1 or die > 6:
        raise ValueError(f"Die must be 1..6, got {die}.")
    if relative_position == BASE:
        return PATH_START if die == 6 else None
    if relative_position == FINISHED:
        return None
    target = relative_position + die
    if target > FINISHED:
        return None
    return target


def bumped_pieces(state: GameState, player: str, target: int) -> Tuple[Tuple[str, int], ...]:
    target_absolute = state_absolute_position(state, player, target)
    if target_absolute is None or target_absolute in SAFE_ABSOLUTE_SQUARES:
        return ()

    bumped: List[Tuple[str, int]] = []
    for opponent in state.active_players:
        if opponent == player:
            continue
        for piece_index, opponent_position in enumerate(state.pieces[opponent]):
            if state_absolute_position(state, opponent, opponent_position) == target_absolute:
                bumped.append((opponent, piece_index))
    return tuple(bumped)


def legal_moves(state: GameState, player: Optional[str] = None, die: Optional[int] = None) -> Tuple[MoveOption, ...]:
    if state.winner is not None:
        return ()
    active_player = player or state.current_player
    if active_player not in state.active_players:
        return ()
    active_die = die if die is not None else state.pending_roll
    if active_die is None:
        return ()

    options: List[MoveOption] = []
    for piece_index, source in enumerate(state.pieces[active_player]):
        target = move_target(source, active_die)
        if target is None:
            continue
        options.append(
            MoveOption(
                player=active_player,
                piece_index=piece_index,
                source=source,
                target=target,
                target_absolute=state_absolute_position(state, active_player, target),
                bumps=bumped_pieces(state, active_player, target),
            )
        )
    return tuple(options)


def roll_die(state: GameState, die: int) -> RollOutcome:
    if state.winner is not None:
        raise ValueError("Cannot roll after the game is over.")
    if state.pending_roll is not None:
        raise ValueError("Resolve the pending move before rolling again.")

    player = state.current_player
    if player not in state.active_players:
        raise ValueError(f"{player_label(player)} is not active in this game.")

    state.stats[player].rolls += 1
    if die == 6:
        state.stats[player].sixes += 1
    _append_log(state, f"{player_label(player)} rolled {die}.")

    if die == 6:
        if state.consecutive_sixes >= 2:
            state.consecutive_sixes = 0
            state.stats[player].passes += 1
            state.stats[player].three_six_losses += 1
            _append_log(state, f"{player_label(player)} rolled three 6s in a row and loses the turn.")
            _advance_turn(state)
            return RollOutcome(player=player, die=die, passed=True, turn_lost=True)
        state.consecutive_sixes += 1
    else:
        state.consecutive_sixes = 0

    options = legal_moves(state, player, die)
    if not options:
        state.stats[player].passes += 1
        _append_log(state, f"{player_label(player)} has no legal move.")
        _advance_turn(state)
        return RollOutcome(player=player, die=die, passed=True)

    state.pending_roll = die
    return RollOutcome(player=player, die=die, legal_moves=options)


def apply_pending_move(state: GameState, piece_index: int) -> MoveResult:
    if state.winner is not None:
        raise ValueError("Cannot move after the game is over.")
    if state.pending_roll is None:
        raise ValueError("There is no pending roll to resolve.")

    player = state.current_player
    die = state.pending_roll
    options = {option.piece_index: option for option in legal_moves(state, player, die)}
    if piece_index not in options:
        raise ValueError(f"{piece_label(player, piece_index)} cannot legally move on a roll of {die}.")

    option = options[piece_index]
    state.pieces[player][piece_index] = option.target
    for bumped_player, bumped_piece_index in option.bumps:
        state.pieces[bumped_player][bumped_piece_index] = BASE
        state.stats[bumped_player].times_bumped += 1

    state.pending_roll = None
    if option.bumps:
        state.consecutive_sixes = 0

    state.stats[player].moves += 1
    state.stats[player].total_pips_moved += 0 if option.is_from_base else die
    state.stats[player].bumps_given += len(option.bumps)
    if option.is_from_base:
        state.stats[player].base_exits += 1
    if option.enters_home_column:
        state.stats[player].home_entries += 1
    if option.lands_on_safe_square or option.target >= HOME_COLUMN_START:
        state.stats[player].safe_landings += 1
    if option.reaches_home_triangle:
        state.stats[player].pieces_finished += 1

    move_text = (
        f"{player_label(player)} moved {piece_label(player, piece_index)} "
        f"from {position_label(player, option.source, state)} to {position_label(player, option.target, state)}."
    )
    _append_log(state, move_text)
    if option.bumps:
        bumped_labels = ", ".join(piece_label(bumped_player, idx) for bumped_player, idx in option.bumps)
        _append_log(state, f"{player_label(player)} bumped {bumped_labels}.")

    winner = player if all(position == FINISHED for position in state.pieces[player]) else None
    if winner is not None:
        state.winner = winner
        _append_log(state, f"{player_label(winner)} wins.")
        return MoveResult(
            player=player,
            piece_index=piece_index,
            die=die,
            source=option.source,
            target=option.target,
            bumps=option.bumps,
            extra_turn=False,
            winner=winner,
        )

    extra_turn = die == 6
    if extra_turn:
        state.stats[player].extra_turns += 1
        _append_log(state, f"{player_label(player)} earned another roll.")
    else:
        _advance_turn(state)

    return MoveResult(
        player=player,
        piece_index=piece_index,
        die=die,
        source=option.source,
        target=option.target,
        bumps=option.bumps,
        extra_turn=extra_turn,
        winner=None,
    )


def position_label(player: str, relative_position: int, state: Optional[GameState] = None) -> str:
    if relative_position == BASE:
        return "base"
    if relative_position == FINISHED:
        return "home triangle"
    if is_in_home_column(relative_position):
        return f"home column {relative_position - HOME_COLUMN_START + 1}"
    absolute = (
        state_absolute_position(state, player, relative_position)
        if state is not None
        else absolute_position(player, relative_position)
    )
    return f"track {relative_position} / abs {absolute}"


def format_move_option(option: MoveOption) -> str:
    label = (
        f"{piece_label(option.player, option.piece_index)}: "
        f"{position_label(option.player, option.source)} -> {position_label(option.player, option.target)}"
    )
    if option.bumps:
        bumped = ", ".join(piece_label(player, idx) for player, idx in option.bumps)
        label += f" | bumps {bumped}"
    if option.reaches_home_triangle:
        label += " | finishes"
    elif option.lands_on_safe_square:
        label += " | safe"
    return label


def pieces_at_absolute(state: GameState, absolute_index: int) -> Tuple[Tuple[str, int], ...]:
    occupants: List[Tuple[str, int]] = []
    for player in state.active_players:
        for piece_index, relative_position in enumerate(state.pieces[player]):
            if state_absolute_position(state, player, relative_position) == absolute_index:
                occupants.append((player, piece_index))
    return tuple(occupants)


def finished_count(state: GameState, player: str) -> int:
    return sum(1 for position in state.pieces[player] if position == FINISHED)


def _advance_turn(state: GameState) -> None:
    state.pending_roll = None
    state.consecutive_sixes = 0
    state.current_player = next_player(state.current_player, state.active_players)
    state.turn_count += 1
    state.stats[state.current_player].turns_started += 1
    _append_log(state, f"Turn passes to {player_label(state.current_player)}.")


def _append_log(state: GameState, message: str) -> None:
    state.log.append(message)
    if len(state.log) > 250:
        del state.log[:50]


def iter_piece_positions(state: GameState) -> Iterable[Tuple[str, int, int]]:
    for player in state.active_players:
        for piece_index, relative_position in enumerate(state.pieces[player]):
            yield player, piece_index, relative_position
