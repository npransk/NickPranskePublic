"""Monte Carlo scenario bank: the "Monte Carlo" half of the stack.

A scenario is one possible future. In every scenario, each of the 13
future turn slots gets its own fixed pool of dice: 5 dice for the opening
roll and 10 more that are used, in order, by rerolls. For every slot and
every category we replay that turn with a policy that chases only that
category (for example, the best keeps for fours, or for a large straight).
Because all targets draw from the same pool, they share the opening roll
and the luck of the rerolls, just as a real player has one set of dice per
turn.

The result is `hands[s, slot, category]`: the final hand you would end the
turn with if you went for `category` in that turn slot of scenario `s`. The
target policies don't depend on the scorecard, so the bank is built once
and reused for every state. Evaluating every candidate state against the
same bank (common random numbers) means comparisons between candidates
are not swamped by sampling noise.
"""

from __future__ import annotations

import numpy as np

from yahtzee_mcmilp.rules import HANDS, KEEPS, N_CATEGORIES, SCORE_TABLE
from yahtzee_mcmilp.turn import TurnPlan

N_SLOTS = 13
POOL = 15  # 5 opening dice + up to 5 + 5 rerolled dice

# Map a count vector to its hand index through a base-6 code (counts are 0..5).
_CODE_WEIGHTS = 6 ** np.arange(6)
_CODE_TO_HAND = np.full(6**6, -1, dtype=np.int16)
_CODE_TO_HAND[HANDS.astype(np.int64) @ _CODE_WEIGHTS] = np.arange(len(HANDS))

TARGET_PLANS = [TurnPlan(SCORE_TABLE[:, c].astype(float)) for c in range(N_CATEGORIES)]


def _counts_of(faces: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Face counts (N, 6) from die faces 0..5 (N, k), counting only where mask is True."""
    onehot = faces[..., None] == np.arange(6)
    return (onehot & mask[..., None]).sum(axis=1)


def play_target_turns(pools: np.ndarray, category: int) -> np.ndarray:
    """Play one turn per row of `pools` (N, 15) chasing `category`; return final hand indices."""
    plan = TARGET_PLANS[category]
    n = len(pools)
    counts = _counts_of(pools[:, :5], np.ones((n, 5), dtype=bool))
    hand = _CODE_TO_HAND[counts @ _CODE_WEIGHTS]
    ptr = np.full(n, 5)
    for rolls_left in (2, 1):
        kept = KEEPS[plan.keep[rolls_left][hand]].astype(np.int64)
        n_reroll = 5 - kept.sum(axis=1)
        idx = np.minimum(ptr[:, None] + np.arange(5), POOL - 1)
        faces = np.take_along_axis(pools, idx, axis=1)
        counts = kept + _counts_of(faces, np.arange(5) < n_reroll[:, None])
        hand = _CODE_TO_HAND[counts @ _CODE_WEIGHTS]
        ptr = ptr + n_reroll
    return hand


class ScenarioBank:
    """Fixed sample of futures; `hands[s, slot, c]` is the hand from chasing c in that slot."""

    def __init__(self, n_scenarios: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        pools = rng.integers(0, 6, size=(n_scenarios * N_SLOTS, POOL))
        hands = np.stack([play_target_turns(pools, c) for c in range(N_CATEGORIES)], axis=1)
        self.n_scenarios = n_scenarios
        self.seed = seed
        self.hands = hands.reshape(n_scenarios, N_SLOTS, N_CATEGORIES)
        # Base scores per (scenario, slot, category) and whether each hand is a Yahtzee.
        cats = np.arange(N_CATEGORIES)
        self.scores = SCORE_TABLE[self.hands, cats].astype(np.int32)
        self.is_yahtzee = HANDS[self.hands].max(axis=-1) == 5
