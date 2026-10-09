"""Bot scaffolding for the Ludo app.

Bot contract:
    choose_move(state, legal_moves, rng) -> piece_index

Return the integer piece index, 0 through 3, for the piece you want to move.
The app validates the returned piece index and falls back to a legal move if a
bot returns something invalid.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Protocol, Sequence

from ludo_engine import (
    BASE,
    BLUE,
    FINISHED,
    GREEN,
    HOME_COLUMN_END,
    HOME_COLUMN_START,
    MoveOption,
    GameState,
    PLAYER_ORDER,
    PLAYER_SPECS,
    RED,
    SAFE_ABSOLUTE_SQUARES,
    YELLOW,
    clone_state,
    state_absolute_position,
    state_relative_index_for_absolute,
)


class LudoBot(Protocol):
    name: str

    def choose_move(
        self,
        state: GameState,
        legal_moves: Sequence[MoveOption],
        rng: random.Random,
    ) -> int:
        """Return the 0-based piece index to move."""


class YellowProbabilityBot:
    """Reserved scaffold for Nick's probability bot.

    Replace choose_move with your probability model. Return one piece index
    from the provided legal_moves list, such as 0 for Y1 or 3 for Y4.
    """

    name = "Nick probability bot placeholder"

    def choose_move(
        self,
        state: GameState,
        legal_moves: Sequence[MoveOption],
        rng: random.Random,
    ) -> int:
        return _solid_default_choice(legal_moves)


class BlueRLBot:
    """Reserved scaffold for Nick's reinforcement learning bot.

    Replace choose_move with your trained policy. The state object uses
    player-relative positions, so a neural policy can consume state.pieces
    directly or transform it into features.
    """

    name = "Nick RL bot placeholder"

    def choose_move(
        self,
        state: GameState,
        legal_moves: Sequence[MoveOption],
        rng: random.Random,
    ) -> int:
        return rng.choice(tuple(legal_moves)).piece_index


class GreenCodexBot:
    """Risk-aware tempo bot for Green.

    The strategy is deliberately transparent: finish pieces, bump opponents,
    get pieces into home, prefer safe landings, and avoid leaving pieces on
    squares that are likely to be captured on the next roll.
    """

    name = "Codex risk-tempo bot"

    def choose_move(
        self,
        state: GameState,
        legal_moves: Sequence[MoveOption],
        rng: random.Random,
    ) -> int:
        scored = [
            (self._score_move(state, option, rng), option.piece_index)
            for option in legal_moves
        ]
        scored.sort(reverse=True)
        return scored[0][1]

    def _score_move(self, state: GameState, option: MoveOption, rng: random.Random) -> float:
        score = rng.uniform(0.0, 0.2)

        if option.reaches_home_triangle:
            score += 10_000
        if option.enters_home_column:
            score += 145
        if HOME_COLUMN_START <= option.target <= HOME_COLUMN_END:
            score += 80
        if option.is_from_base:
            score += 55
        if option.bumps:
            score += 260 * len(option.bumps)
        if option.lands_on_safe_square:
            score += 70

        score += max(option.target, 0) * 1.8
        score += max(option.target - max(option.source, 0), 0) * 4.0

        target_absolute = option.target_absolute
        if target_absolute is not None:
            own_stack = _own_stack_size(state, option.player, target_absolute)
            score += own_stack * 18

            target_threat = _capture_threat_count(state, option.player, target_absolute)
            if target_threat:
                stack_penalty = max(1, own_stack + 1)
                score -= 62 * target_threat * stack_penalty

        source_absolute = state_absolute_position(state, option.player, option.source)
        if source_absolute is not None:
            source_threat = _capture_threat_count(state, option.player, source_absolute)
            target_threat = (
                _capture_threat_count(state, option.player, target_absolute)
                if target_absolute is not None
                else 0
            )
            if source_threat > target_threat:
                score += 42 * (source_threat - target_threat)

        if _is_last_piece_in_base(state, option):
            score -= 12

        return score


BOT_REGISTRY: Dict[str, LudoBot] = {
    YELLOW: YellowProbabilityBot(),
    BLUE: BlueRLBot(),
    GREEN: GreenCodexBot(),
}


def choose_bot_piece(
    player: str,
    state: GameState,
    options: Sequence[MoveOption],
    rng: random.Random,
) -> int:
    if not options:
        raise ValueError("A bot cannot choose from an empty legal move list.")

    bot = BOT_REGISTRY[player]
    legal_piece_indices = {option.piece_index for option in options}
    try:
        choice = int(bot.choose_move(clone_state(state), tuple(options), rng))
    except Exception:
        choice = _solid_default_choice(options)

    if choice in legal_piece_indices:
        return choice
    return _solid_default_choice(options)


def bot_name(player: str) -> str:
    bot = BOT_REGISTRY.get(player)
    if bot is None:
        return "Human player"
    return bot.name


def choose_default_piece(options: Sequence[MoveOption]) -> int:
    return _solid_default_choice(options)


def _solid_default_choice(options: Sequence[MoveOption]) -> int:
    ranked = sorted(
        options,
        key=lambda option: (
            option.reaches_home_triangle,
            len(option.bumps),
            option.enters_home_column,
            option.is_from_base,
            option.target,
        ),
        reverse=True,
    )
    return ranked[0].piece_index


def _own_stack_size(state: GameState, player: str, absolute_index: int) -> int:
    count = 0
    for position in state.pieces[player]:
        if state_absolute_position(state, player, position) == absolute_index:
            count += 1
    return count


def _capture_threat_count(state: GameState, player: str, target_absolute: Optional[int]) -> int:
    if target_absolute is None or target_absolute in SAFE_ABSOLUTE_SQUARES:
        return 0

    threats = 0
    for opponent in state.active_players:
        if opponent == player:
            continue
        opponent_target_relative = state_relative_index_for_absolute(state, opponent, target_absolute)
        for opponent_position in state.pieces[opponent]:
            if opponent_position < 0 or opponent_position > 51:
                continue
            distance = opponent_target_relative - opponent_position
            if 1 <= distance <= 6:
                threats += 1
    return threats


def _is_last_piece_in_base(state: GameState, option: MoveOption) -> bool:
    if not option.is_from_base:
        return False
    return sum(1 for position in state.pieces[option.player] if position == BASE) == 1
