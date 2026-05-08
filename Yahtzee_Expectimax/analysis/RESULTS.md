# Yahtzee High-Score Bot Results

Generated on 2026-05-08.

## Goal

Explore what I would build if the main goal were to average very high Yahtzee scores, around 240+.

## Techniques Tried

| Bot | Technique | Games | Mean | Median | Max | Over 200 | Over 240 | Over 250 | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboPolicy | Fast hand-tuned heuristic | 10,000 | 214.73 | 204.00 | 600 | 54.19% | 24.68% | 18.54% | 16.65s |
| TournamentPolicy | Exact within-turn search + tuned scorecard utility | 200 | 236.39 | 230.50 | 526 | 75.00% | 45.00% | 36.00% | 168.52s |
| Numba turn-local search | Compiled exact turn search, default params | 200 | 228.72 | 219.00 | 586 | 72.00% | 35.50% | 29.00% | 6.92s |
| Numba search best sample | Random-searched utility params | 800 | 240.56 | 232.00 | 620 | 77.13% | 44.50% | 36.63% | 4.53s |
| Numba refined validation | Tuned params, independent validation | 10,000 | 236.26 | 224.00 | 618 | 77.02% | 39.18% | 32.63% | 55.88s |

## Interpretation

The best validated result in this folder is still in the 236 range. A smaller random-search sample crossed 240, but a larger independent validation regressed to 236.26, so I would not honestly claim this project has a stable 240+ policy yet.

The first exact dynamic-programming solver was the theoretically right direction, but the naive Python implementation was too slow because full scorecard-state expectimax has a large cache. The practical next step is to make that solver table-based and compiled with NumPy/Numba/Rust, or persist the dynamic-programming table once built.

This pass added a Numba bottom-up turn evaluator and parameter search. That made optimization practical, but it is still only turn-local: it chooses the best action for the current turn using a tuned scorecard utility, not an exact value for every remaining scorecard state. That is why it plateaus below the true optimal range.

## What I Would Do Next To Break 240+

1. Use full scorecard-state dynamic programming with a compact value table:
   - `used_mask`
   - upper subtotal capped at 63
   - Yahtzee bonus state
   - dice-count state
   - rolls remaining

2. Precompute:
   - dice outcome distributions
   - legal keep masks for every dice count
   - category scores for every dice count
   - transition lists for every keep

3. Compile the inner loops with Numba or Rust. The current `numba_solver.py` is a good starting point for the dice/transition tables.

4. Save the resulting policy/value table to disk so gameplay is instant.

5. Use the exact solver as the teacher for the MLOps neural bot:
   - Generate millions of state/action examples from the solver.
   - Train a neural net to imitate the exact policy.
   - Use the neural net in Streamlit for instant moves.

That approach should be able to exceed 240 and likely approach known near-optimal Yahtzee averages.
