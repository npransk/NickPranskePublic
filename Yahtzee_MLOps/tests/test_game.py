import random

from yahtzee_mlops.game import STATE_SIZE, YahtzeeGame


def test_state_vector_size_is_stable() -> None:
    game = YahtzeeGame()
    assert game.state_vector().shape == (STATE_SIZE,)


def test_first_action_is_roll_all() -> None:
    game = YahtzeeGame()
    assert game.legal_actions() == [("roll", (0, 0, 0, 0, 0, 0))]


def test_roll_then_score_advances_turn() -> None:
    game = YahtzeeGame(rng=random.Random(1))
    game.step(("roll", (0, 0, 0, 0, 0, 0)))
    score_before = game.total_score
    game.step(("score", "chance"))
    assert game.turn == 1
    assert game.roll_count == 0
    assert game.total_score >= score_before


def test_keep_counts_cannot_keep_missing_dice() -> None:
    game = YahtzeeGame(rng=random.Random(1))
    game.dice = (1, 1, 2, 3, 4)
    game.roll_count = 1
    try:
        game.roll((0, 0, 0, 0, 5, 0))
    except ValueError:
        assert True
    else:
        raise AssertionError("Expected invalid keep counts to raise.")
