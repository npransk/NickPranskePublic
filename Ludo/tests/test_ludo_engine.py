from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ludo_engine import (  # noqa: E402
    BASE,
    BLUE,
    FINISHED,
    GREEN,
    HOME_COLUMN_START,
    RED,
    YELLOW,
    apply_pending_move,
    create_new_game,
    legal_moves,
    roll_die,
)


class LudoEngineTest(unittest.TestCase):
    def test_piece_needs_six_to_leave_base(self) -> None:
        state = create_new_game()

        self.assertEqual(legal_moves(state, RED, 5), ())
        options = legal_moves(state, RED, 6)

        self.assertEqual({option.piece_index for option in options}, {0, 1, 2, 3})
        self.assertTrue(all(option.target == 0 for option in options))

    def test_exact_throw_required_to_finish(self) -> None:
        state = create_new_game()
        state.pieces[RED] = [56, BASE, BASE, BASE]

        self.assertEqual(legal_moves(state, RED, 2), ())
        options = legal_moves(state, RED, 1)

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0].target, FINISHED)

    def test_safe_start_square_cannot_be_bumped(self) -> None:
        state = create_new_game()
        state.pieces[RED] = [12, BASE, BASE, BASE]
        state.pieces[YELLOW] = [0, BASE, BASE, BASE]

        roll_die(state, 1)
        apply_pending_move(state, 0)

        self.assertEqual(state.pieces[RED][0], 13)
        self.assertEqual(state.pieces[YELLOW][0], 0)

    def test_unsafe_landing_bumps_all_opposing_pieces(self) -> None:
        state = create_new_game()
        state.pieces[RED] = [5, BASE, BASE, BASE]
        state.pieces[YELLOW] = [47, 47, BASE, BASE]

        roll_die(state, 3)
        apply_pending_move(state, 0)

        self.assertEqual(state.pieces[RED][0], 8)
        self.assertEqual(state.pieces[YELLOW][0], BASE)
        self.assertEqual(state.pieces[YELLOW][1], BASE)

    def test_third_six_loses_turn_without_moving(self) -> None:
        state = create_new_game()

        roll_die(state, 6)
        apply_pending_move(state, 0)
        roll_die(state, 6)
        apply_pending_move(state, 0)
        roll_die(state, 6)

        self.assertEqual(state.pieces[RED][0], 6)
        self.assertEqual(state.current_player, YELLOW)
        self.assertIsNone(state.pending_roll)

    def test_bump_resets_consecutive_six_counter_but_keeps_extra_turn(self) -> None:
        state = create_new_game()
        state.pieces[RED] = [0, BASE, BASE, BASE]
        state.pieces[YELLOW] = [45, BASE, BASE, BASE]

        roll_die(state, 6)
        apply_pending_move(state, 0)

        self.assertEqual(state.pieces[YELLOW][0], BASE)
        self.assertEqual(state.consecutive_sixes, 0)
        self.assertEqual(state.current_player, RED)

        roll_die(state, 6)

        self.assertEqual(state.consecutive_sixes, 1)
        self.assertEqual(state.current_player, RED)

    def test_home_column_is_private_and_safe(self) -> None:
        state = create_new_game()
        state.pieces[RED] = [HOME_COLUMN_START, BASE, BASE, BASE]
        state.pieces[GREEN] = [13, BASE, BASE, BASE]

        self.assertEqual(legal_moves(state, GREEN, 1)[0].target, 14)

    def test_first_player_to_finish_all_four_wins(self) -> None:
        state = create_new_game()
        state.pieces[RED] = [FINISHED, FINISHED, FINISHED, 56]

        roll_die(state, 1)
        result = apply_pending_move(state, 3)

        self.assertEqual(result.winner, RED)
        self.assertEqual(state.winner, RED)

    def test_inactive_players_are_skipped_in_turn_order(self) -> None:
        state = create_new_game([RED, GREEN])

        self.assertEqual(state.current_player, RED)
        roll_die(state, 1)

        self.assertEqual(state.current_player, GREEN)
        self.assertEqual(state.active_players, (RED, GREEN))

    def test_bump_stats_are_recorded(self) -> None:
        state = create_new_game([RED, YELLOW, BLUE])
        state.pieces[RED] = [5, BASE, BASE, BASE]
        state.pieces[YELLOW] = [47, BASE, BASE, BASE]

        roll_die(state, 3)
        apply_pending_move(state, 0)

        self.assertEqual(state.stats[RED].bumps_given, 1)
        self.assertEqual(state.stats[YELLOW].times_bumped, 1)
        self.assertEqual(state.stats[RED].moves, 1)

    def test_two_players_are_assigned_opposite_start_squares(self) -> None:
        state = create_new_game([RED, YELLOW])

        self.assertEqual(state.start_indices[RED], 0)
        self.assertEqual(state.start_indices[YELLOW], 26)


if __name__ == "__main__":
    unittest.main()
