from itertools import permutations

import numpy as np
import pytest

from yahtzee_mcmilp.agent import GreedyAgent, MilpAgent
from yahtzee_mcmilp.milp import (
    MilpValueEstimator, _assignment_value, _bounded_assignment, _build_block, _solve_blocks,
)
from yahtzee_mcmilp.rules import (
    FIRST_ROLL, IS_YAHTZEE, SCORE_TABLE, TRANSITION, State, hand_index,
)
from yahtzee_mcmilp.scenarios import ScenarioBank
from yahtzee_mcmilp.simulate import run_games
from yahtzee_mcmilp.turn import TurnPlan


def test_scores():
    assert SCORE_TABLE[hand_index([2, 2, 5, 5, 6]), 4] == 10
    assert SCORE_TABLE[hand_index([3, 3, 3, 5, 5]), 8] == 25
    assert SCORE_TABLE[hand_index([1, 2, 3, 4, 6]), 9] == 30
    assert SCORE_TABLE[hand_index([1, 2, 3, 4, 6]), 10] == 0
    assert SCORE_TABLE[hand_index([6, 6, 6, 6, 6]), 11] == 50
    assert SCORE_TABLE[hand_index([6, 6, 6, 6, 2]), 7] == 26


def test_probabilities():
    assert np.allclose(TRANSITION.sum(axis=1), 1.0)
    assert FIRST_ROLL[IS_YAHTZEE].sum() == pytest.approx(6 / 6**5)


def test_turn_solver_known_values():
    # Optimal Chance-only play averages 23.33; best P(Yahtzee) in one turn is ~4.60%.
    assert TurnPlan(SCORE_TABLE[:, 12].astype(float)).expected == pytest.approx(70 / 3)
    assert TurnPlan(IS_YAHTZEE.astype(float)).expected == pytest.approx(0.04603, abs=1e-4)


def test_state_bonuses():
    yz = hand_index([4] * 5)
    _, s = State().apply(11, yz)
    assert s.ybonus
    points, _ = s.apply(3, yz)
    assert points == 20 + 100
    points, s2 = State(upper=60).apply(5, hand_index([6, 6, 1, 2, 3]))
    assert points == 12 + 35 and s2.upper == 63


def _brute_force(w, upper_pts, deficit, y_col, y_hit, bonus_ok):
    n = len(w)
    return max(_assignment_value(np.array(p), w, upper_pts, deficit, y_col, y_hit, bonus_ok)
               for p in permutations(range(n)))


@pytest.mark.parametrize("seed", range(25))
def test_milp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 6
    w = rng.integers(0, 30, size=(n, n)).astype(float)
    upper_pts = w * (np.arange(n) < 3)
    deficit = int(rng.integers(0, 60))
    y_col = n - 1
    y_hit = rng.random(n) < 0.4
    bonus_ok = rng.random((n, n)) < 0.3
    bonus_ok[:, y_col] = False
    expected = _brute_force(w, upper_pts, deficit, y_col, y_hit, bonus_ok)
    got = _solve_blocks([_build_block(w, upper_pts, deficit, y_col, y_hit, bonus_ok)])
    assert got == pytest.approx(expected)
    value, exact = _bounded_assignment(w, upper_pts, deficit, y_col, y_hit, bonus_ok)
    assert value <= expected + 1e-9
    if exact:
        assert value == pytest.approx(expected)


def test_value_estimates():
    est = MilpValueEstimator(ScenarioBank(16, seed=1))
    assert est.value(State(used=(1 << 13) - 1)) == 0.0
    # One category left (chance): its hindsight value is the average Chance-chasing roll.
    last = State(used=((1 << 13) - 1) & ~(1 << 12))
    assert 20 < est.value(last) < 28
    assert est.value(State()) > 240


def test_agents_play_full_games():
    small = MilpValueEstimator(ScenarioBank(8, seed=3))
    for agent in (GreedyAgent(), MilpAgent(small)):
        result = run_games(agent, games=2, seed=7)
        assert result["games"] == 2 and result["min"] > 0
