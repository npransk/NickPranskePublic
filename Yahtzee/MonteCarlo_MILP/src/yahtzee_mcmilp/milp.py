"""MILP value estimator: the "MILP" half of the stack.

To value a scorecard state, take each Monte Carlo scenario (see
scenarios.py) and ask: knowing how every remaining turn would go if I
chased each open category, what is the best way to assign turns to
categories? That is an assignment problem with two bonus couplings that
make it a MILP:

    maximise  sum_tc w[t,c] x[t,c] + 35 b + 100 sum z[t,c]
    s.t.      each remaining turn fills exactly one open category
              each open category is filled by exactly one turn
              upper points assigned >= (63 - current upper) * b          (upper bonus)
              z[t,c] <= x[t,c]                                           (Yahtzee bonus:
              z[t,c] <= sum_{t' < t} [turn t' rolled a Yahtzee] x[t',Y]   needs an earlier
              x, b, z binary                                              50 in the box)

The state's value is the average optimum over scenarios: a sample average
approximation (SAA) of "hindsight optimisation". Solving with knowledge of
future dice is optimistic, because a real player can't see the future. So
these values are upper-biased, which the agent corrects for with `discount`
(see agent.py).

Most scenarios never need the MILP solver. A plain assignment
(scipy's linear_sum_assignment, microseconds) is exact whenever the bonus
constraints can't change the answer: the unconstrained optimum already
earns the upper bonus, or the bonus is out of reach. Only the remaining
scenarios go to HiGHS, batched into one block-diagonal MILP per state.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix

from yahtzee_mcmilp.rules import UPPER_BONUS, UPPER_TARGET, YAHTZEE, YAHTZEE_BONUS, State
from yahtzee_mcmilp.scenarios import ScenarioBank


@contextmanager
def _quiet_stdout():
    """HiGHS's MIP code occasionally prints debug lines straight to fd 1; hide them."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


@dataclass
class SolveStats:
    states: int = 0
    scenarios: int = 0
    assignment_only: int = 0
    milp_scenarios: int = 0
    milp_calls: int = 0
    cache_hits: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class _Block:
    """One scenario's MILP, in local variable numbering."""

    c: list[float] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    vals: list[float] = field(default_factory=list)
    lo: list[float] = field(default_factory=list)
    hi: list[float] = field(default_factory=list)

    def var(self, cost: float) -> int:
        self.c.append(cost)
        return len(self.c) - 1

    def row(self, entries: list[tuple[int, float]], lo: float, hi: float) -> None:
        r = len(self.lo)
        for col, val in entries:
            self.rows.append(r)
            self.cols.append(col)
            self.vals.append(val)
        self.lo.append(lo)
        self.hi.append(hi)


def _build_block(w: np.ndarray, upper_pts: np.ndarray, deficit: int,
                 y_col: int | None, y_hit: np.ndarray | None, bonus_ok: np.ndarray | None) -> _Block:
    n = w.shape[0]
    blk = _Block()
    x = [[blk.var(float(w[t, k])) for k in range(n)] for t in range(n)]
    for t in range(n):
        blk.row([(x[t][k], 1.0) for k in range(n)], 1.0, 1.0)
    for k in range(n):
        blk.row([(x[t][k], 1.0) for t in range(n)], 1.0, 1.0)
    if deficit > 0:
        b = blk.var(float(UPPER_BONUS))
        entries = [(x[t][k], float(upper_pts[t, k])) for t in range(n) for k in range(n) if upper_pts[t, k]]
        blk.row(entries + [(b, -float(deficit))], 0.0, np.inf)
    if y_col is not None:
        for t in range(n):
            earlier = [(x[tp][y_col], -1.0) for tp in range(t) if y_hit[tp]]
            for k in range(n):
                if k == y_col or not bonus_ok[t, k] or not earlier:
                    continue
                z = blk.var(float(YAHTZEE_BONUS))
                blk.row([(z, 1.0), (x[t][k], -1.0)], -np.inf, 0.0)
                blk.row([(z, 1.0)] + earlier, -np.inf, 0.0)
    return blk


def _assignment_value(cols: np.ndarray, w: np.ndarray, upper_pts: np.ndarray, deficit: int,
                      y_col: int | None, y_hit: np.ndarray | None, bonus_ok: np.ndarray | None) -> float:
    """True objective (with both bonuses) of the assignment turn t -> category cols[t]."""
    rows = np.arange(len(cols))
    value = float(w[rows, cols].sum())
    if deficit and upper_pts[rows, cols].sum() >= deficit:
        value += UPPER_BONUS
    if y_hit is not None:
        t_y = int(np.flatnonzero(cols == y_col)[0])
        if y_hit[t_y]:
            later = rows > t_y
            value += YAHTZEE_BONUS * float(bonus_ok[rows[later], cols[later]].sum())
    return value


def _bounded_assignment(w, upper_pts, deficit, y_col, y_hit, bonus_ok) -> tuple[float, bool]:
    """Try to solve a scenario with plain assignments only.

    Upper bound: assignment where every possible bonus is paid out (+35 if the
    upper target is reachable at all). Lower bound: the true value of a few
    candidate assignments. If they meet, that is the MILP optimum and no
    solver call is needed. Returns (value, exact).
    """
    candidates = [linear_sum_assignment(w, maximize=True)[1]]
    w_ub = w if y_hit is None else w + YAHTZEE_BONUS * bonus_ok
    if y_hit is not None:
        candidates.append(linear_sum_assignment(w_ub, maximize=True)[1])
    ub = float(w_ub[np.arange(len(w)), candidates[-1]].sum())
    if deficit:
        cols_u = linear_sum_assignment(upper_pts, maximize=True)[1]
        if upper_pts[np.arange(len(w)), cols_u].sum() >= deficit:
            ub += UPPER_BONUS
            candidates.append(cols_u)
    lb = max(_assignment_value(c, w, upper_pts, deficit, y_col, y_hit, bonus_ok) for c in candidates)
    return lb, lb >= ub - 1e-9


def _solve_blocks(blocks: list[_Block]) -> float:
    """Solve independent scenario MILPs as one block-diagonal MILP; return total optimum."""
    c, rows, cols, vals, lo, hi = [], [], [], [], [], []
    v_off = r_off = 0
    for blk in blocks:
        c.extend(blk.c)
        rows.extend(r + r_off for r in blk.rows)
        cols.extend(v + v_off for v in blk.cols)
        vals.extend(blk.vals)
        lo.extend(blk.lo)
        hi.extend(blk.hi)
        v_off += len(blk.c)
        r_off += len(blk.lo)
    a = coo_matrix((vals, (rows, cols)), shape=(r_off, v_off)).tocsr()
    with _quiet_stdout():
        res = milp(-np.array(c), constraints=LinearConstraint(a, lo, hi),
                   integrality=np.ones(v_off), bounds=Bounds(0, 1),
                   options={"presolve": False})  # presolve costs more than it saves at this size
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return -res.fun


class MilpValueEstimator:
    """Estimated future points from a scorecard state, via Monte Carlo scenarios + MILP."""

    def __init__(self, bank: ScenarioBank | None = None):
        self.bank = bank or ScenarioBank()
        self.cache: dict[tuple[int, int, bool], float] = {}
        self.stats = SolveStats()

    def value(self, state: State) -> float:
        key = state.key()
        if key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[key]
        v = self._estimate(state)
        self.cache[key] = v
        return v

    def _estimate(self, state: State) -> float:
        open_cats = state.open_categories()
        n = len(open_cats)
        if n == 0:
            return 0.0
        self.stats.states += 1
        bank = self.bank
        cats = np.array(open_cats)
        scores = bank.scores[:, :n, :][:, :, cats]          # (S, n, n)
        is_y = bank.is_yahtzee[:, :n, :][:, :, cats]        # (S, n, n)
        upper_mask = cats < 6
        upper_pts = scores * upper_mask                      # upper points per (s, t, k)

        deficit = UPPER_TARGET - state.upper if upper_mask.any() and state.upper < UPPER_TARGET else 0
        y_col = open_cats.index(YAHTZEE) if YAHTZEE in open_cats and not state.ybonus else None
        not_y = cats != YAHTZEE
        weights = scores.astype(float)
        if state.ybonus:
            weights = weights + YAHTZEE_BONUS * (is_y & not_y)

        total = 0.0
        pending: list[_Block] = []
        for s in range(bank.n_scenarios):
            self.stats.scenarios += 1
            w = weights[s]
            y_hit = bonus_ok = None
            if y_col is not None:
                y_hit = is_y[s, :, y_col]
                bonus_ok = is_y[s] & not_y
                # A bonus is possible only if some Yahtzee lands after an earlier Yahtzee-box Yahtzee.
                first_hit = int(np.argmax(y_hit)) if y_hit.any() else n
                if not bonus_ok[first_hit + 1:].any():
                    y_hit = bonus_ok = None
            value, exact = _bounded_assignment(w, upper_pts[s], deficit, y_col, y_hit, bonus_ok)
            if exact:
                total += value
                self.stats.assignment_only += 1
                continue
            pending.append(_build_block(w, upper_pts[s], deficit, y_col if y_hit is not None else None,
                                        y_hit, bonus_ok))
        if pending:
            self.stats.milp_scenarios += len(pending)
            self.stats.milp_calls += 1
            total += _solve_blocks(pending)
        return total / bank.n_scenarios
