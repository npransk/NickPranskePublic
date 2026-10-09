from __future__ import annotations

import html
import random
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import streamlit as st

from bots import bot_name, choose_bot_piece, choose_default_piece
from ludo_engine import (
    BASE,
    BLUE,
    FINISHED,
    GREEN,
    HOME_COLUMN_START,
    PLAYER_ORDER,
    PLAYER_SPECS,
    RED,
    YELLOW,
    GameState,
    MoveOption,
    PlayerStats,
    apply_pending_move,
    create_new_game,
    finished_count,
    format_move_option,
    is_in_home_column,
    is_on_shared_track,
    iter_piece_positions,
    legal_moves,
    piece_label,
    player_label,
    roll_die,
    state_absolute_position,
)


TRACK_COORDS: Tuple[Tuple[int, int], ...] = (
    (13, 6),
    (12, 6),
    (11, 6),
    (10, 6),
    (9, 6),
    (8, 5),
    (8, 4),
    (8, 3),
    (8, 2),
    (8, 1),
    (8, 0),
    (7, 0),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 3),
    (6, 4),
    (6, 5),
    (5, 6),
    (4, 6),
    (3, 6),
    (2, 6),
    (1, 6),
    (0, 6),
    (0, 7),
    (0, 8),
    (1, 8),
    (2, 8),
    (3, 8),
    (4, 8),
    (5, 8),
    (6, 9),
    (6, 10),
    (6, 11),
    (6, 12),
    (6, 13),
    (6, 14),
    (7, 14),
    (8, 14),
    (8, 13),
    (8, 12),
    (8, 11),
    (8, 10),
    (8, 9),
    (9, 8),
    (10, 8),
    (11, 8),
    (12, 8),
    (13, 8),
    (14, 8),
    (14, 7),
    (14, 6),
)

START_COORDS = {player: TRACK_COORDS[spec.start_index] for player, spec in PLAYER_SPECS.items()}
START_COORD_TO_PLAYER = {coord: player for player, coord in START_COORDS.items()}
START_INDEX_TO_SEAT = {
    spec.start_index: player
    for player, spec in PLAYER_SPECS.items()
}

HOME_COORDS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    RED: ((13, 7), (12, 7), (11, 7), (10, 7), (9, 7)),
    YELLOW: ((7, 1), (7, 2), (7, 3), (7, 4), (7, 5)),
    BLUE: ((1, 7), (2, 7), (3, 7), (4, 7), (5, 7)),
    GREEN: ((7, 13), (7, 12), (7, 11), (7, 10), (7, 9)),
}

TOKEN_COLORS = {
    RED: "#d92d20",
    YELLOW: "#d19b00",
    BLUE: "#2563eb",
    GREEN: "#15803d",
}

DIE_FACE_LABELS = {
    1: "⚀",
    2: "⚁",
    3: "⚂",
    4: "⚃",
    5: "⚄",
    6: "⚅",
}

SVG_WIDTH = 980
SVG_HEIGHT = 720
CELL_SIZE = 30
CELL_STEP = 34
BOARD_X = 245
BOARD_Y = 82
DOCK_WIDTH = 174
DOCK_HEIGHT = 142

DOCK_RECTS = {
    YELLOW: (46, 58, DOCK_WIDTH, DOCK_HEIGHT),
    BLUE: (760, 58, DOCK_WIDTH, DOCK_HEIGHT),
    RED: (46, 470, DOCK_WIDTH, DOCK_HEIGHT),
    GREEN: (760, 470, DOCK_WIDTH, DOCK_HEIGHT),
}

DOCK_SLOT_OFFSETS = ((58, 56), (116, 56), (58, 108), (116, 108))


def main() -> None:
    st.set_page_config(page_title="Ludo Arena", layout="wide")
    _ensure_session()
    _handle_board_query_action()
    _inject_css()
    _render_sidebar()

    play_tab, sim_tab = st.tabs(["Play", "Simulation Lab"])

    with play_tab:
        top_left_col, _ = st.columns([0.32, 1], gap="large")
        with top_left_col:
            st.checkbox("Auto-roll Red", key="auto_roll_human")
        st.title("Ludo Arena")
        state = st.session_state.game
        board_col, control_col = st.columns([1.35, 1], gap="large")
        with board_col:
            _render_board(state)
        with control_col:
            _render_scoreboard(state)
            st.divider()
            _render_turn_controls(state)
            st.divider()
            _render_current_stats(state)
            st.divider()
            _render_log(state)

    with sim_tab:
        _render_simulation_lab()

    _continue_human_auto_roll_after_render()
    _continue_bot_turn_after_render()


def _ensure_session() -> None:
    if "seed" not in st.session_state:
        st.session_state.seed = random.randint(1, 999_999)
    if "active_players" not in st.session_state:
        st.session_state.active_players = PLAYER_ORDER
    if "game" not in st.session_state or _game_needs_reset(st.session_state.game):
        _reset_game(st.session_state.seed, st.session_state.active_players)
    if "last_roll" not in st.session_state:
        st.session_state.last_roll = None
    if "last_roll_player" not in st.session_state:
        st.session_state.last_roll_player = None
    if "animate_die" not in st.session_state:
        st.session_state.animate_die = False
    if "last_move" not in st.session_state:
        st.session_state.last_move = None
    if "bot_watch_active" not in st.session_state:
        st.session_state.bot_watch_active = False
    if "bot_watch_remaining" not in st.session_state:
        st.session_state.bot_watch_remaining = 0
    if "auto_roll_human" not in st.session_state:
        st.session_state.auto_roll_human = False


def _game_needs_reset(state: object) -> bool:
    return not all(
        hasattr(state, attribute)
        for attribute in ("active_players", "start_indices", "stats", "pieces", "current_player")
    )


def _reset_game(seed: int, active_players: Sequence[str]) -> None:
    normalized_players = _selected_players(active_players)
    st.session_state.seed = int(seed)
    st.session_state.active_players = normalized_players
    st.session_state.rng = random.Random(int(seed))
    st.session_state.game = create_new_game(normalized_players)
    st.session_state.last_roll = None
    st.session_state.last_roll_player = None
    st.session_state.animate_die = False
    st.session_state.last_move = None
    st.session_state.bot_watch_active = False
    st.session_state.bot_watch_remaining = 0


def _handle_board_query_action() -> None:
    state = st.session_state.game
    if state.winner is not None:
        _clear_interaction_query()
        return

    move_piece = _query_value("move_piece")
    if move_piece is not None:
        _clear_interaction_query()
        if _is_human_turn(state) and state.pending_roll is not None:
            try:
                piece_index = int(move_piece)
            except ValueError:
                piece_index = -1
            legal_piece_indices = {option.piece_index for option in legal_moves(state, state.current_player, state.pending_roll)}
            if piece_index in legal_piece_indices:
                _apply_move_with_animation(state, piece_index)
        _rerun()


def _query_value(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _clear_interaction_query() -> None:
    if _query_value("move_piece") is not None:
        st.query_params.clear()


def _roll_for_current_player(state: GameState):
    die = st.session_state.rng.randint(1, 6)
    st.session_state.last_roll = die
    st.session_state.last_roll_player = state.current_player
    st.session_state.animate_die = True
    return roll_die(state, die)


def _apply_move_with_animation(state: GameState, piece_index: int):
    die = state.pending_roll
    options = {option.piece_index: option for option in legal_moves(state, state.current_player, die)}
    option = options[piece_index]
    source_key = _location_key(state, option.player, option.piece_index, option.source)
    target_key = _location_key(state, option.player, option.piece_index, option.target)
    result = apply_pending_move(state, piece_index)
    st.session_state.last_move = {
        "player": option.player,
        "piece_index": option.piece_index,
        "source_key": source_key,
        "target_key": target_key,
        "die": die,
        "turn_count": state.turn_count,
        "bumps": option.bumps,
        "label": f"{piece_label(option.player, option.piece_index)} moved {die}",
    }
    return result


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Game Setup")
        seed = st.number_input("Seed", min_value=1, max_value=999_999_999, value=int(st.session_state.seed), step=1)

        st.subheader("Active Players")
        selected_players = []
        for player in PLAYER_ORDER:
            default = player in st.session_state.active_players
            if st.checkbox(player_label(player), value=default, key=f"active_{player}"):
                selected_players.append(player)

        if len(selected_players) < 2:
            st.warning("Select at least two players.")

        if st.button("New Game With Selection", use_container_width=True, disabled=len(selected_players) < 2):
            _reset_game(int(seed), selected_players)
            _rerun()

        state = st.session_state.game
        if tuple(selected_players) != state.active_players:
            st.caption("Start a new game to apply the checked player set.")

        st.caption("Red is manual when active. Bot turns advance automatically with short animation pauses.")
        for player in PLAYER_ORDER:
            if player == RED:
                continue
            st.caption(f"{player_label(player)}: {bot_name(player)}")


def _render_scoreboard(state: GameState) -> None:
    st.subheader("Scoreboard")
    for player in state.active_players:
        finished = finished_count(state, player)
        positions = ", ".join(
            piece_label(player, idx) + "=" + _compact_position(position)
            for idx, position in enumerate(state.pieces[player])
        )
        marker = " active" if player == state.current_player and state.winner is None else ""
        st.markdown(
            f"<div class='score-row score-{player.lower()}'>"
            f"<strong>{html.escape(player_label(player))}</strong>{html.escape(marker)}"
            f"<span>{finished}/4 home</span>"
            f"<small>{html.escape(positions)}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_turn_controls(state: GameState) -> None:
    st.subheader("Turn")
    if state.winner is not None:
        st.success(f"{player_label(state.winner)} wins.")
        return

    current_player = state.current_player
    st.markdown(f"**{player_label(current_player)}**")
    _render_die_panel(state)
    if _is_human_turn(state):
        _render_human_controls(state)
    else:
        st.caption(bot_name(current_player))
        st.caption("Bot turns advance automatically.")


def _render_human_controls(state: GameState) -> None:
    if state.pending_roll is None:
        roll_col, value_col = st.columns([0.75, 1.25], gap="small")
        with roll_col:
            if st.button("Roll", type="primary", use_container_width=True):
                _roll_for_current_player(state)
                _rerun()
        with value_col:
            _render_roll_value_badge()
        return

    options = legal_moves(state, RED, state.pending_roll)
    st.caption("Legal pieces and landing squares are highlighted on the board. Click a highlighted piece to move it.")
    for option in options:
        if st.button(format_move_option(option), key=_move_key(option, state.turn_count), use_container_width=True):
            _apply_move_with_animation(state, option.piece_index)
            _rerun()


def _render_roll_value_badge() -> None:
    last_roll = st.session_state.last_roll
    last_roll_player = st.session_state.last_roll_player
    if last_roll is None or last_roll_player != RED:
        label = "Roll waiting"
        die = "?"
        number = "?"
    else:
        label = f"Red rolled {last_roll}"
        die = DIE_FACE_LABELS[last_roll]
        number = str(last_roll)
    st.markdown(
        f"<div class='roll-value-badge'>"
        f"<span class='roll-value-die'>{html.escape(die)}</span>"
        f"<span><strong>{html.escape(number)}</strong><small>{html.escape(label)}</small></span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_die_panel(state: GameState) -> None:
    last_roll = st.session_state.last_roll
    last_roll_player = st.session_state.last_roll_player
    display_die = state.pending_roll or (last_roll if last_roll_player == state.current_player else last_roll)
    rolling = bool(st.session_state.animate_die)
    player = state.current_player
    die_content = DIE_FACE_LABELS.get(display_die, "?")
    die_number = str(display_die) if display_die else "?"
    caption = (
        f"{player_label(last_roll_player)} rolled {last_roll}"
        if last_roll and last_roll_player
        else "Waiting for roll"
    )
    rolling_class = " die-rolling" if rolling else ""
    st.markdown(
        f"<div class='die-panel die-panel-{player.lower()}'>"
        "<div class='die-link'>"
        f"<span class='die-face die-face-{player.lower()}{rolling_class}'>"
        f"<span class='die-symbol'>{html.escape(die_content)}</span>"
        f"<span class='die-number'>{html.escape(die_number)}</span>"
        "</span>"
        "</div>"
        f"<span class='die-caption'>{html.escape(caption)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.session_state.animate_die = False


def _render_current_stats(state: GameState) -> None:
    st.subheader("Current Stats")
    _render_table(_stats_rows_for_state(state), "current-stats")


def _render_log(state: GameState) -> None:
    st.subheader("Move Log")
    for entry in reversed(state.log[-12:]):
        st.write(entry)


def _render_simulation_lab() -> None:
    st.title("Simulation Lab")
    st.caption("Runs automated games using the active player selection from the sidebar. Red uses a neutral fallback policy during simulations.")

    selected_players = _sidebar_selected_players()
    if len(selected_players) < 2:
        st.warning("Select at least two active players in the sidebar before simulating.")
        return

    setup_col, result_col = st.columns([0.85, 1.35], gap="large")
    with setup_col:
        st.subheader("Batch")
        game_count = st.number_input("Games", min_value=1, max_value=1000, value=100, step=25)
        sim_seed = st.number_input("Simulation Seed", min_value=1, max_value=999_999_999, value=int(st.session_state.seed), step=1)
        max_actions = st.number_input("Max Actions Per Game", min_value=200, max_value=10_000, value=3000, step=100)
        show_last_log = st.checkbox("Show Last Game Log", value=True)

        if st.button("Run Simulations", type="primary", use_container_width=True):
            st.session_state.simulation_summary = _run_simulations(
                selected_players,
                int(sim_seed),
                int(game_count),
                int(max_actions),
            )

    with result_col:
        summary = st.session_state.get("simulation_summary")
        if not summary:
            st.info("Run a batch to generate win rates, bump stats, pass counts, and tempo stats.")
            return

        metric_cols = st.columns(4)
        metric_cols[0].metric("Games", summary["game_count"])
        metric_cols[1].metric("Completed", summary["completed_games"])
        metric_cols[2].metric("Timed Out", summary["timed_out_games"])
        metric_cols[3].metric("Avg Actions", f"{summary['avg_actions']:.1f}")

        st.subheader("Player Summary")
        _render_table(summary["player_rows"], "player-summary")

        st.subheader("Game Outcomes")
        _render_table(summary["game_rows"], "game-outcomes", max_rows=250)

        if show_last_log:
            st.subheader("Last Game Log")
            for entry in reversed(summary["last_log"][-30:]):
                st.write(entry)


def _play_one_automated_action() -> None:
    state = st.session_state.game
    if state.winner is not None or _is_human_turn(state):
        return

    acting_player = state.current_player
    if state.pending_roll is None:
        _roll_for_current_player(state)
        return

    if state.winner is not None or state.current_player != acting_player or state.pending_roll is None:
        return

    options = legal_moves(state, acting_player, state.pending_roll)
    if not options:
        return
    piece_index = _choose_automated_piece(acting_player, state, options, st.session_state.rng)
    _apply_move_with_animation(state, piece_index)


def _continue_bot_turn_after_render() -> None:
    state = st.session_state.game
    if state.winner is not None or _is_human_turn(state):
        st.session_state.bot_watch_active = False
        st.session_state.bot_watch_remaining = 0
        return

    if st.session_state.bot_watch_remaining <= 0:
        st.session_state.bot_watch_remaining = 3000

    delay = 0.85 if state.pending_roll is not None else 0.55
    time.sleep(delay)
    _play_one_automated_action()
    st.session_state.bot_watch_remaining -= 1
    _rerun()


def _continue_human_auto_roll_after_render() -> None:
    if not st.session_state.auto_roll_human:
        return

    state = st.session_state.game
    if state.winner is not None or not _is_human_turn(state) or state.pending_roll is not None:
        return

    time.sleep(0.45)
    _roll_for_current_player(state)
    _rerun()


def _simulate_game(active_players: Sequence[str], seed: int, max_actions: int) -> GameState:
    state = create_new_game(active_players)
    rng = random.Random(seed)
    actions = 0
    while state.winner is None and actions < max_actions:
        acting_player = state.current_player
        if state.pending_roll is None:
            roll_die(state, rng.randint(1, 6))

        if state.pending_roll is not None and state.current_player == acting_player:
            options = legal_moves(state, acting_player, state.pending_roll)
            if options:
                piece_index = _choose_automated_piece(acting_player, state, options, rng)
                apply_pending_move(state, piece_index)
        actions += 1

    if state.winner is None:
        state.log.append(f"Simulation stopped after {max_actions} actions without a winner.")
    return state


def _run_simulations(active_players: Sequence[str], seed: int, game_count: int, max_actions: int) -> Dict[str, object]:
    aggregate = {
        player: {
            "wins": 0,
            "rolls": 0,
            "moves": 0,
            "passes": 0,
            "extra_turns": 0,
            "three_six_losses": 0,
            "bumps_given": 0,
            "times_bumped": 0,
            "base_exits": 0,
            "home_entries": 0,
            "safe_landings": 0,
            "pieces_finished": 0,
            "total_pips_moved": 0,
        }
        for player in active_players
    }
    game_rows: List[Dict[str, object]] = []
    completed_games = 0
    total_actions = 0
    last_state = create_new_game(active_players)

    for game_index in range(game_count):
        game_seed = seed + game_index
        state = _simulate_game(active_players, game_seed, max_actions)
        last_state = state
        total_actions += state.turn_count
        if state.winner is not None:
            completed_games += 1
            aggregate[state.winner]["wins"] += 1

        for player in active_players:
            stats = state.stats[player]
            player_aggregate = aggregate[player]
            player_aggregate["rolls"] += stats.rolls
            player_aggregate["moves"] += stats.moves
            player_aggregate["passes"] += stats.passes
            player_aggregate["extra_turns"] += stats.extra_turns
            player_aggregate["three_six_losses"] += stats.three_six_losses
            player_aggregate["bumps_given"] += stats.bumps_given
            player_aggregate["times_bumped"] += stats.times_bumped
            player_aggregate["base_exits"] += stats.base_exits
            player_aggregate["home_entries"] += stats.home_entries
            player_aggregate["safe_landings"] += stats.safe_landings
            player_aggregate["pieces_finished"] += finished_count(state, player)
            player_aggregate["total_pips_moved"] += stats.total_pips_moved

        game_rows.append(
            {
                "Game": game_index + 1,
                "Seed": game_seed,
                "Winner": player_label(state.winner) if state.winner else "Timed out",
                "Actions": state.turn_count,
                "Bumps": sum(state.stats[player].bumps_given for player in active_players),
                "Passes": sum(state.stats[player].passes for player in active_players),
            }
        )

    player_rows = []
    for player in active_players:
        player_aggregate = aggregate[player]
        wins = int(player_aggregate["wins"])
        bumps_given = int(player_aggregate["bumps_given"])
        times_bumped = int(player_aggregate["times_bumped"])
        player_rows.append(
            {
                "Player": player_label(player),
                "Wins": wins,
                "Win %": round(100 * wins / game_count, 1),
                "Avg Rolls": round(player_aggregate["rolls"] / game_count, 1),
                "Avg Moves": round(player_aggregate["moves"] / game_count, 1),
                "Bumped Others": bumps_given,
                "Got Bumped": times_bumped,
                "Bump Diff": bumps_given - times_bumped,
                "Passes": player_aggregate["passes"],
                "Three-6 Losses": player_aggregate["three_six_losses"],
                "Base Exits": player_aggregate["base_exits"],
                "Home Entries": player_aggregate["home_entries"],
                "Avg Finished": round(player_aggregate["pieces_finished"] / game_count, 2),
                "Avg Pips": round(player_aggregate["total_pips_moved"] / game_count, 1),
            }
        )

    return {
        "active_players": tuple(active_players),
        "game_count": game_count,
        "completed_games": completed_games,
        "timed_out_games": game_count - completed_games,
        "avg_actions": total_actions / game_count,
        "player_rows": player_rows,
        "game_rows": game_rows,
        "last_log": list(last_state.log),
    }


def _choose_automated_piece(
    player: str,
    state: GameState,
    options: Sequence[MoveOption],
    rng: random.Random,
) -> int:
    if player == RED:
        return choose_default_piece(options)
    return choose_bot_piece(player, state, options, rng)


def _is_human_turn(state: GameState) -> bool:
    return state.current_player == RED and RED in state.active_players and state.winner is None


def _render_board(state: GameState) -> None:
    source_options, target_options = _board_option_maps(state)
    parts = [
        "<div class='ludo-board-wrap'>",
        f"<svg class='ludo-cross-board' viewBox='0 0 {SVG_WIDTH} {SVG_HEIGHT}' role='img' aria-label='Ludo cross board'>",
        "<defs>",
        "<filter id='softShadow' x='-20%' y='-20%' width='140%' height='140%'>",
        "<feDropShadow dx='0' dy='10' stdDeviation='12' flood-color='#0f172a' flood-opacity='0.16'/>",
        "</filter>",
        "</defs>",
        "<rect class='board-bg' x='8' y='8' width='964' height='704' rx='28'/>",
    ]

    for player in state.active_players:
        parts.append(_dock_svg(state, player, source_options))

    parts.extend(_track_cells_svg(state, target_options))
    parts.extend(_home_cells_svg(state, target_options))
    parts.append(_center_svg(state, target_options))
    parts.append(_move_trail_svg())
    parts.extend(_token_svgs(state, source_options))
    parts.append("</svg></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _board_option_maps(state: GameState) -> Tuple[Dict[Tuple[object, ...], MoveOption], Dict[Tuple[object, ...], MoveOption]]:
    source_options: Dict[Tuple[object, ...], MoveOption] = {}
    target_options: Dict[Tuple[object, ...], MoveOption] = {}
    if state.winner is not None or state.pending_roll is None:
        return source_options, target_options

    for option in legal_moves(state, state.current_player, state.pending_roll):
        source_key = _location_key(state, option.player, option.piece_index, option.source)
        target_key = _location_key(state, option.player, option.piece_index, option.target)
        if source_key is not None:
            source_options[source_key] = option
        if target_key is not None:
            target_options[target_key] = option
    return source_options, target_options


def _move_trail_svg() -> str:
    last_move = st.session_state.get("last_move")
    if not last_move:
        return ""
    source_key = last_move.get("source_key")
    target_key = last_move.get("target_key")
    if source_key is None or target_key is None:
        return ""

    source_x, source_y = _location_center(source_key)
    target_x, target_y = _location_center(target_key)
    player = str(last_move.get("player"))
    label = str(last_move.get("label", "Last move"))
    return (
        f"<g class='move-trail move-trail-{player.lower()}'>"
        f"<title>{html.escape(label)}</title>"
        f"<line x1='{source_x}' y1='{source_y}' x2='{target_x}' y2='{target_y}'/>"
        f"<circle class='move-ghost' cx='{source_x}' cy='{source_y}' r='12'/>"
        f"<circle class='move-target-pulse' cx='{target_x}' cy='{target_y}' r='15'/>"
        "</g>"
    )


def _is_last_moved_token(player: str, piece_index: int) -> bool:
    last_move = st.session_state.get("last_move")
    return bool(
        last_move
        and last_move.get("player") == player
        and last_move.get("piece_index") == piece_index
    )


def _dock_svg(state: GameState, player: str, source_options: Dict[Tuple[object, ...], MoveOption]) -> str:
    seat = _player_visual_seat(state, player)
    x, y, width, height = DOCK_RECTS[seat]
    slots = []
    for idx, (offset_x, offset_y) in enumerate(DOCK_SLOT_OFFSETS):
        cx = x + offset_x
        cy = y + offset_y
        source_key = ("base", player, idx)
        legal_class = ""
        if source_key in source_options:
            legal_class = f" legal-source legal-{player.lower()}"
        slots.append(
            f"<circle class='dock-slot dock-slot-{player.lower()}{legal_class}' cx='{cx}' cy='{cy}' r='18'/>"
            f"<text class='dock-slot-label' x='{cx}' y='{cy + 4}'>{idx + 1}</text>"
        )
    return (
        f"<g class='dock dock-{player.lower()}'>"
        f"<rect x='{x}' y='{y}' width='{width}' height='{height}' rx='24'/>"
        f"<text class='dock-label' x='{x + 22}' y='{y + 30}'>{html.escape(player_label(player))}</text>"
        f"{''.join(slots)}"
        "</g>"
    )


def _track_cells_svg(
    state: GameState,
    target_options: Dict[Tuple[object, ...], MoveOption],
) -> List[str]:
    active_start_players = {
        state.start_indices.get(player, PLAYER_SPECS[player].start_index): player
        for player in state.active_players
    }
    cells = []
    for absolute_index, coord in enumerate(TRACK_COORDS):
        location_key = ("track", absolute_index)
        start_player = active_start_players.get(absolute_index)
        classes = ["svg-cell", "track-cell"]
        label = ""
        if start_player:
            classes.extend(["start-cell", f"start-{start_player.lower()}"])
            label = PLAYER_SPECS[start_player].short_name
        if location_key in target_options:
            classes.extend(["legal-destination", f"legal-{target_options[location_key].player.lower()}"])
        cells.append(_cell_svg(coord, classes, label, target_options.get(location_key)))
    return cells


def _home_cells_svg(
    state: GameState,
    target_options: Dict[Tuple[object, ...], MoveOption],
) -> List[str]:
    cells = []
    for player in state.active_players:
        seat = _player_visual_seat(state, player)
        coords = HOME_COORDS[seat]
        for index, coord in enumerate(coords):
            location_key = ("home", player, index)
            classes = ["svg-cell", "home-cell", f"home-{player.lower()}"]
            if location_key in target_options:
                classes.extend(["legal-destination", f"legal-{target_options[location_key].player.lower()}"])
            cells.append(_cell_svg(coord, classes, str(index + 1), target_options.get(location_key)))
    return cells


def _center_svg(state: GameState, target_options: Dict[Tuple[object, ...], MoveOption]) -> str:
    cx, cy = _cell_center((7, 7))
    finished_text = " ".join(
        f"{PLAYER_SPECS[player].short_name}{finished_count(state, player)}"
        for player in state.active_players
    )
    finishing_options = [
        option
        for key, option in target_options.items()
        if key and key[0] == "finished"
    ]
    classes = ["finish-center"]
    if finishing_options:
        classes.extend(["legal-finish", f"legal-{finishing_options[0].player.lower()}"])
    return (
        f"<g class='{html.escape(' '.join(classes))}'>"
        f"<path d='M {cx} {cy - 58} L {cx + 58} {cy} L {cx} {cy + 58} L {cx - 58} {cy} Z'/>"
        f"<text x='{cx}' y='{cy - 4}'>HOME</text>"
        f"<text class='finish-counts' x='{cx}' y='{cy + 18}'>{html.escape(finished_text)}</text>"
        "</g>"
    )


def _cell_svg(
    coord: Tuple[int, int],
    classes: Sequence[str],
    label: str = "",
    option: MoveOption | None = None,
) -> str:
    row, col = coord
    x, y = _cell_xy(row, col)
    label_svg = ""
    if label:
        label_svg = f"<text class='cell-label' x='{x + CELL_SIZE / 2}' y='{y + CELL_SIZE / 2 + 4}'>{html.escape(label)}</text>"
    title_svg = ""
    if option is not None:
        title_svg = f"<title>{html.escape(piece_label(option.player, option.piece_index))} can land here</title>"
    return (
        f"<g class='cell-group'>"
        f"<rect class='{html.escape(' '.join(classes))}' x='{x}' y='{y}' width='{CELL_SIZE}' height='{CELL_SIZE}' rx='8'/>"
        f"{title_svg}"
        f"{label_svg}"
        "</g>"
    )


def _token_svgs(
    state: GameState,
    source_options: Dict[Tuple[object, ...], MoveOption],
) -> List[str]:
    token_groups = _token_locations(state)
    tokens = []
    for location_key, occupants in token_groups.items():
        cx, cy = _location_center(location_key)
        visible_occupants = sorted(occupants, key=lambda item: (PLAYER_ORDER.index(item[0]), item[1]))[:4]
        extra_count = max(0, len(occupants) - len(visible_occupants))
        offsets = _stack_offsets(len(visible_occupants) + (1 if extra_count else 0))
        for offset_index, (player, piece_index) in enumerate(visible_occupants):
            dx, dy = offsets[offset_index]
            source_key = _location_key(state, player, piece_index, state.pieces[player][piece_index])
            option = source_options.get(source_key)
            moving = _is_last_moved_token(player, piece_index)
            tokens.append(_token_svg(player, piece_index, cx + dx, cy + dy, option, moving))
        if extra_count:
            dx, dy = offsets[-1]
            tokens.append(
                f"<g class='token token-extra'>"
                f"<circle cx='{cx + dx}' cy='{cy + dy}' r='11'/>"
                f"<text x='{cx + dx}' y='{cy + dy + 4}'>+{extra_count}</text>"
                "</g>"
            )
    return tokens


def _token_locations(state: GameState) -> Dict[Tuple[object, ...], List[Tuple[str, int]]]:
    locations: Dict[Tuple[object, ...], List[Tuple[str, int]]] = {}
    for player, piece_index, relative_position in iter_piece_positions(state):
        location_key = _location_key(state, player, piece_index, relative_position)
        if location_key is None:
            continue
        locations.setdefault(location_key, []).append((player, piece_index))
    return locations


def _location_key(state: GameState, player: str, piece_index: int, relative_position: int) -> Tuple[object, ...] | None:
    if relative_position == BASE:
        return ("base", player, piece_index)
    if relative_position == FINISHED:
        return ("finished", player)
    if is_on_shared_track(relative_position):
        absolute = state_absolute_position(state, player, relative_position)
        return ("track", absolute) if absolute is not None else None
    if is_in_home_column(relative_position):
        return ("home", player, relative_position - HOME_COLUMN_START)
    return None


def _location_center(location_key: Tuple[object, ...]) -> Tuple[float, float]:
    state = st.session_state.game
    kind = location_key[0]
    if kind == "base":
        player = str(location_key[1])
        piece_index = int(location_key[2])
        seat = _player_visual_seat(state, player)
        x, y, _, _ = DOCK_RECTS[seat]
        offset_x, offset_y = DOCK_SLOT_OFFSETS[piece_index]
        return x + offset_x, y + offset_y
    if kind == "track":
        absolute_index = int(location_key[1])
        return _cell_center(TRACK_COORDS[absolute_index])
    if kind == "home":
        player = str(location_key[1])
        home_index = int(location_key[2])
        seat = _player_visual_seat(state, player)
        return _cell_center(HOME_COORDS[seat][home_index])
    if kind == "finished":
        player = str(location_key[1])
        return _finish_center(player)
    raise ValueError(f"Unknown token location: {location_key}")


def _token_svg(
    player: str,
    piece_index: int,
    cx: float,
    cy: float,
    option: MoveOption | None = None,
    moving: bool = False,
) -> str:
    label = piece_label(player, piece_index)
    classes = ["token", f"token-{player.lower()}"]
    if option is not None:
        classes.extend(["legal-token", f"legal-{player.lower()}"])
    if moving:
        classes.append("token-moving")
    token_markup = (
        f"<g class='{html.escape(' '.join(classes))}'>"
        f"<title>{html.escape(label)}</title>"
        f"<circle cx='{cx}' cy='{cy}' r='11'/>"
        f"<text x='{cx}' y='{cy + 4}'>{html.escape(label)}</text>"
        "</g>"
    )
    if option is not None and st.session_state.game.current_player == RED and _is_human_turn(st.session_state.game):
        href = f"?move_piece={piece_index}&amp;turn={st.session_state.game.turn_count}"
        return f"<a class='token-link' href='{href}' target='_self' aria-label='Move {html.escape(label)}'>{token_markup}</a>"
    return (
        token_markup
    )


def _cell_xy(row: int, col: int) -> Tuple[int, int]:
    return BOARD_X + col * CELL_STEP, BOARD_Y + row * CELL_STEP


def _cell_center(coord: Tuple[int, int]) -> Tuple[float, float]:
    row, col = coord
    x, y = _cell_xy(row, col)
    return x + CELL_SIZE / 2, y + CELL_SIZE / 2


def _finish_center(player: str) -> Tuple[float, float]:
    cx, cy = _cell_center((7, 7))
    offsets = {
        RED: (0, 32),
        YELLOW: (-32, 0),
        BLUE: (0, -32),
        GREEN: (32, 0),
    }
    dx, dy = offsets[player]
    return cx + dx, cy + dy


def _player_visual_seat(state: GameState, player: str) -> str:
    start_index = state.start_indices.get(player, PLAYER_SPECS[player].start_index)
    return START_INDEX_TO_SEAT.get(start_index, player)


def _stack_offsets(count: int) -> Tuple[Tuple[int, int], ...]:
    if count <= 1:
        return ((0, 0),)
    if count == 2:
        return ((-7, 0), (7, 0))
    if count == 3:
        return ((-8, -6), (8, -6), (0, 8))
    return ((-8, -8), (8, -8), (-8, 8), (8, 8))


def _stats_rows_for_state(state: GameState) -> List[Dict[str, object]]:
    return [
        _stats_row(player, state.stats[player], finished_count(state, player))
        for player in state.active_players
    ]


def _stats_row(player: str, stats: PlayerStats, finished: int) -> Dict[str, object]:
    return {
        "Player": player_label(player),
        "Rolls": stats.rolls,
        "Moves": stats.moves,
        "Passes": stats.passes,
        "Bumped Others": stats.bumps_given,
        "Got Bumped": stats.times_bumped,
        "Extra Rolls": stats.extra_turns,
        "Three-6 Losses": stats.three_six_losses,
        "Base Exits": stats.base_exits,
        "Home Entries": stats.home_entries,
        "Finished": finished,
        "Pips": stats.total_pips_moved,
    }


def _render_table(rows: Sequence[Dict[str, object]], table_class: str, max_rows: int | None = None) -> None:
    if not rows:
        st.caption("No rows yet.")
        return

    displayed_rows = list(rows[:max_rows] if max_rows is not None else rows)
    headers = list(displayed_rows[0].keys())
    head_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_html = []
    for row in displayed_rows:
        cells = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        body_html.append(f"<tr>{cells}</tr>")

    note = ""
    if max_rows is not None and len(rows) > max_rows:
        note = f"<p class='table-note'>Showing {max_rows} of {len(rows)} games.</p>"

    st.markdown(
        f"<div class='table-shell {html.escape(table_class)}'>"
        f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(body_html)}</tbody></table>"
        f"</div>{note}",
        unsafe_allow_html=True,
    )


def _sidebar_selected_players() -> Tuple[str, ...]:
    return _selected_players(
        player
        for player in PLAYER_ORDER
        if st.session_state.get(f"active_{player}", player in st.session_state.active_players)
    )


def _selected_players(players: Iterable[str]) -> Tuple[str, ...]:
    selected = set(players)
    return tuple(player for player in PLAYER_ORDER if player in selected)


def _compact_position(position: int) -> str:
    if position == BASE:
        return "base"
    if position == FINISHED:
        return "home"
    if position >= HOME_COLUMN_START:
        return f"h{position - HOME_COLUMN_START + 1}"
    return f"t{position}"


def _move_key(option: MoveOption, turn_count: int) -> str:
    return f"move-{turn_count}-{option.player}-{option.piece_index}-{option.source}-{option.target}"


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 18% 12%, rgba(37, 99, 235, 0.16), transparent 28%),
                radial-gradient(circle at 88% 72%, rgba(21, 128, 61, 0.14), transparent 26%),
                #0b1120;
            color: #e5e7eb;
        }

        main h1,
        main h2,
        main h3,
        main p,
        main label,
        main [data-testid="stMarkdownContainer"] {
            color: #e5e7eb;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] {
            background: #080d19;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }

        [data-testid="stTabs"] button {
            color: #e5e7eb;
        }

        div[data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 0.75rem;
        }

        .die-panel {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin: 0.5rem 0 0.85rem;
            padding: 0.78rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.76);
        }

        .die-link {
            text-decoration: none;
            cursor: default;
        }

        .die-face {
            width: 66px;
            height: 66px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            position: relative;
            color: #f8fafc;
            background: linear-gradient(145deg, #f8fafc, #dbe4f0);
            border: 2px solid rgba(255, 255, 255, 0.9);
            box-shadow: 0 16px 30px rgba(0, 0, 0, 0.36), inset 0 -6px 12px rgba(15, 23, 42, 0.16);
        }

        .die-symbol {
            color: #0f172a;
            font-size: 42px;
            font-weight: 900;
            line-height: 1;
        }

        .die-number {
            position: absolute;
            right: -8px;
            bottom: -8px;
            min-width: 24px;
            height: 24px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0 0.25rem;
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid rgba(255, 255, 255, 0.72);
            font-size: 0.82rem;
            font-weight: 900;
        }

        .die-face-red .die-number { background: #d92d20; }
        .die-face-yellow .die-number { background: #a66f00; }
        .die-face-blue .die-number { background: #2563eb; }
        .die-face-green .die-number { background: #15803d; }

        .die-rolling {
            animation: dieRoll 720ms cubic-bezier(0.2, 0.8, 0.2, 1);
        }

        .die-caption {
            color: #cbd5e1;
            font-weight: 750;
        }

        .roll-value-badge {
            min-height: 42px;
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.38rem 0.62rem;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.74);
        }

        .roll-value-die {
            width: 34px;
            height: 34px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background: #f8fafc;
            color: #0f172a;
            font-size: 24px;
            line-height: 1;
        }

        .roll-value-badge strong {
            display: block;
            color: #f8fafc;
            line-height: 1.1;
        }

        .roll-value-badge small {
            display: block;
            color: #94a3b8;
            line-height: 1.15;
        }

        .ludo-board-wrap {
            width: min(100%, 980px);
            margin: 0 auto 1rem;
        }

        .ludo-cross-board {
            width: 100%;
            height: auto;
            display: block;
        }

        .board-bg {
            fill: #111827;
            filter: url(#softShadow);
            stroke: rgba(148, 163, 184, 0.22);
            stroke-width: 1;
        }

        .svg-cell {
            stroke: rgba(226, 232, 240, 0.42);
            stroke-width: 1.8;
            vector-effect: non-scaling-stroke;
        }

        .track-cell {
            fill: #1f2937;
        }

        .home-cell {
            stroke-width: 1.6;
        }

        .home-red { fill: #7f1d1d; }
        .home-yellow { fill: #713f12; }
        .home-blue { fill: #1e3a8a; }
        .home-green { fill: #14532d; }

        .start-cell {
            stroke: #f8fafc;
            stroke-width: 3;
        }

        .start-red { fill: #dc2626; }
        .start-yellow { fill: #ca8a04; }
        .start-blue { fill: #2563eb; }
        .start-green { fill: #16a34a; }

        .legal-source,
        .legal-destination {
            animation: legalPulse 1.2s ease-in-out infinite;
        }

        .legal-red {
            stroke: #f87171 !important;
            filter: drop-shadow(0 0 8px rgba(248, 113, 113, 0.8));
        }

        .legal-yellow {
            stroke: #facc15 !important;
            filter: drop-shadow(0 0 8px rgba(250, 204, 21, 0.76));
        }

        .legal-blue {
            stroke: #60a5fa !important;
            filter: drop-shadow(0 0 8px rgba(96, 165, 250, 0.78));
        }

        .legal-green {
            stroke: #4ade80 !important;
            filter: drop-shadow(0 0 8px rgba(74, 222, 128, 0.78));
        }

        .inactive-cell {
            fill: #111827;
            stroke: #475569;
            opacity: 0.48;
        }

        .cell-label {
            fill: rgba(226, 232, 240, 0.74);
            font-size: 11px;
            font-weight: 800;
            text-anchor: middle;
            pointer-events: none;
        }

        .dock rect {
            stroke-width: 2;
            stroke: rgba(226, 232, 240, 0.16);
        }

        .dock-red rect { fill: #291316; }
        .dock-yellow rect { fill: #2a2110; }
        .dock-blue rect { fill: #111d36; }
        .dock-green rect { fill: #102617; }

        .dock-label {
            font-size: 18px;
            font-weight: 850;
            letter-spacing: 0;
            fill: #f8fafc;
        }

        .dock-slot {
            fill: rgba(15, 23, 42, 0.8);
            stroke-width: 2;
            stroke: rgba(226, 232, 240, 0.22);
        }

        .dock-slot-red { stroke: #d92d20; }
        .dock-slot-yellow { stroke: #d19b00; }
        .dock-slot-blue { stroke: #2563eb; }
        .dock-slot-green { stroke: #15803d; }

        .dock-slot-label {
            fill: rgba(226, 232, 240, 0.48);
            text-anchor: middle;
            font-size: 11px;
            font-weight: 800;
        }

        .finish-center path {
            fill: #0f172a;
            stroke: rgba(226, 232, 240, 0.54);
            stroke-width: 2;
            vector-effect: non-scaling-stroke;
        }

        .finish-center text {
            fill: #f8fafc;
            text-anchor: middle;
            font-size: 14px;
            font-weight: 900;
        }

        .finish-center .finish-counts {
            font-size: 11px;
            fill: #cbd5e1;
        }

        .legal-finish path {
            animation: legalPulse 1.2s ease-in-out infinite;
        }

        .move-trail {
            pointer-events: none;
        }

        .move-trail line {
            stroke-width: 5;
            stroke-linecap: round;
            opacity: 0;
            animation: trailFlash 900ms ease-out;
        }

        .move-trail-red line,
        .move-trail-red circle { stroke: #f87171; fill: rgba(248, 113, 113, 0.2); }
        .move-trail-yellow line,
        .move-trail-yellow circle { stroke: #facc15; fill: rgba(250, 204, 21, 0.2); }
        .move-trail-blue line,
        .move-trail-blue circle { stroke: #60a5fa; fill: rgba(96, 165, 250, 0.2); }
        .move-trail-green line,
        .move-trail-green circle { stroke: #4ade80; fill: rgba(74, 222, 128, 0.2); }

        .move-ghost,
        .move-target-pulse {
            stroke-width: 3;
            opacity: 0;
            animation: targetPulse 900ms ease-out;
        }

        .token circle {
            stroke: rgba(255, 255, 255, 0.95);
            stroke-width: 2;
            filter: drop-shadow(0 2px 3px rgba(15, 23, 42, 0.28));
        }

        .token text {
            fill: #ffffff;
            text-anchor: middle;
            font-size: 8.5px;
            font-weight: 900;
            pointer-events: none;
        }

        .token-red circle { fill: #d92d20; }
        .token-yellow circle { fill: #a66f00; }
        .token-blue circle { fill: #2563eb; }
        .token-green circle { fill: #15803d; }
        .token-extra circle { fill: #111827; }

        .token-link {
            cursor: pointer;
            text-decoration: none;
        }

        .legal-token circle {
            stroke-width: 3;
        }

        .token-link:hover .token {
            transform: translateY(-3px);
        }

        .token {
            transform-box: fill-box;
            transform-origin: center;
            transition: transform 150ms ease;
        }

        .token-moving {
            animation: tokenHop 800ms cubic-bezier(0.2, 0.8, 0.2, 1);
        }

        .score-row {
            border-left: 6px solid #94a3b8;
            background: rgba(17, 24, 39, 0.92);
            color: #e5e7eb;
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.45rem;
            border-top: 1px solid rgba(148, 163, 184, 0.12);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.22);
        }

        .score-row strong {
            display: inline-block;
            margin-right: 0.35rem;
            color: #f8fafc;
        }

        .score-row span {
            float: right;
            color: #cbd5e1;
            font-weight: 700;
        }

        .score-row small {
            display: block;
            clear: both;
            color: #94a3b8;
            margin-top: 0.25rem;
            overflow-wrap: anywhere;
        }

        .score-red { border-left-color: #d92d20; }
        .score-yellow { border-left-color: #d19b00; }
        .score-blue { border-left-color: #2563eb; }
        .score-green { border-left-color: #15803d; }

        .table-shell {
            width: 100%;
            max-height: 430px;
            overflow: auto;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.72);
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.2);
        }

        .table-shell table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
        }

        .table-shell th,
        .table-shell td {
            padding: 0.52rem 0.62rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
            text-align: left;
            white-space: nowrap;
        }

        .table-shell th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #111827;
            color: #f8fafc;
            font-weight: 800;
        }

        .table-shell td {
            color: #dbe4f0;
        }

        .table-shell tr:nth-child(even) td {
            background: rgba(30, 41, 59, 0.34);
        }

        .table-note {
            color: #94a3b8 !important;
            font-size: 0.85rem;
            margin-top: 0.4rem;
        }

        @keyframes dieRoll {
            0% { transform: rotate(0deg) scale(1); }
            18% { transform: rotate(38deg) scale(1.08); }
            36% { transform: rotate(-32deg) scale(0.96); }
            58% { transform: rotate(24deg) scale(1.06); }
            78% { transform: rotate(-10deg) scale(1.02); }
            100% { transform: rotate(0deg) scale(1); }
        }

        @keyframes legalPulse {
            0%, 100% { stroke-width: 3; opacity: 0.9; }
            50% { stroke-width: 5; opacity: 1; }
        }

        @keyframes trailFlash {
            0% { opacity: 0; stroke-dasharray: 1 460; }
            18% { opacity: 0.94; }
            70% { opacity: 0.78; stroke-dasharray: 460 1; }
            100% { opacity: 0; stroke-dasharray: 460 1; }
        }

        @keyframes targetPulse {
            0% { opacity: 0.85; transform: scale(0.5); }
            70% { opacity: 0.35; transform: scale(1.45); }
            100% { opacity: 0; transform: scale(1.65); }
        }

        @keyframes tokenHop {
            0% { transform: scale(0.88); }
            28% { transform: translateY(-9px) scale(1.16); }
            62% { transform: translateY(2px) scale(0.98); }
            100% { transform: translateY(0) scale(1); }
        }

        @media (max-width: 820px) {
            .dock-label {
                font-size: 15px;
            }

            .token text {
                font-size: 7.5px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
