# Yahtzee: Monte Carlo + MILP

A fourth take on the Yahtzee bot. It sits alongside the Dueling DQN in `../Train_Model` and the two attempts in `../Vibe_Code_Attempts`. This version doesn't learn a policy by trial and error. It plans each move by combining three pieces:

| Layer | What it does | Where |
| --- | --- | --- |
| Exact turn solver | Solves the three-roll turn exactly by backward induction over 252 hands and 462 keeps, in under a millisecond. | `turn.py` |
| Monte Carlo scenarios | Samples a fixed bank of possible futures. For each future turn, it records the hand you'd end up with if you chased each category. | `scenarios.py` |
| MILP | For each scenario, finds the best way to assign remaining turns to open categories, including the upper and Yahtzee bonuses. Solved with the open-source HiGHS solver through `scipy.optimize.milp`. | `milp.py` |

At the start of every turn the agent values each way the turn could end:

```
terminal[hand] = max over open categories c of  points(c, hand) + discount * V_milp(scorecard after scoring c)
```

It then plays the turn exactly against those values (`agent.py`). The only thing that differs from the "greedy" baseline is `V_milp`, the long-term value of the scorecard.

## The MILP

For a scorecard with `n` open categories, each of the `S` scenarios gives a score `w[t,c]` for turn `t` spent chasing category `c`. The program is:

```
maximise   sum w[t,c] x[t,c]  +  35 b  +  100 sum z[t,c]
subject to sum_c x[t,c] = 1                      each remaining turn fills one box
           sum_t x[t,c] = 1                      each open box gets one turn
           sum_{c upper} w[t,c] x[t,c] >= (63 - upper_so_far) * b      upper bonus
           z[t,c] <= x[t,c]                                            Yahtzee bonus, only after
           z[t,c] <= sum_{t' < t, turn t' rolled Yahtzee} x[t',Yahtzee] a 50 is in the Yahtzee box
           x, b, z binary
```

Without the bonus rows this is a plain assignment problem, and `linear_sum_assignment` solves that exactly in microseconds. The code first brackets each scenario between an upper and a lower bound using assignments only. Only scenarios where the bonuses actually change the answer go to HiGHS, about 10% of them, batched into one block-diagonal MILP per state. Those cases do need integer programming: the LP relaxation came out fractional in roughly 40% of a sample of them.

The state value is the average optimum over scenarios. This is sample-average approximation of *hindsight optimisation*. Every candidate state is scored against the same scenario bank (common random numbers), so comparisons between moves aren't swamped by sampling noise.

### The known weakness: hindsight is optimistic

The MILP sees the future dice, which a real player can't. For a fresh scorecard it estimates about 320 points, while true optimal play under these rules averages about 254. This is the classic clairvoyance bias of hindsight optimisation and perfect-information Monte Carlo. Because the bias affects every candidate move in the same direction, the agent still makes good relative choices. `--discount` scales the future value down to partly correct for it.

## Results

All runs use the rules below. The ± figure is the standard error of the mean.

| Bot | Games | Mean | Median | Over 200 | Over 250 | Time / game (4 cores) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dueling DQN (`../Train_Model`) | 100 | 168.9 | 173 | ~40% | ~0% | instant |
| Expectimax turn-local + tuned utility (`../Vibe_Code_Attempts`) | 10,000 | 236.3 | 224 | 77.0% | 32.6% | fast |
| **Greedy**: exact turn solver, no look-ahead | 2,000 | 217.6 ± 1.1 | 208 | 62.8% | 19.8% | 0.01 s |
| **MC + MILP**, discount 0.9 | 200 | 244.7 ± 3.9 | 239 | 81.0% | 43.0% | ~2 s |
| **MC + MILP**, discount 1.0 | 200 | 252.5 ± 4.3 | 245 | 85.0% | 45.5% | ~2 s |
| **MC + MILP**, discount 1.0, independent validation (seed 90210) | 1,000 | **247.3 ± 2.0** | 239 | 81.7% | 39.5% | ~1.3 s |

Takeaways:
- The future-value model is worth about 30 points over greedy play with the same exact turn solver.
- The validated 247 beats the DQN by about 78 points and the earlier tuned-utility search by about 11. It is within a few points of what optimal play scores under these rules.
- Shrinking the hindsight values (discount 0.9) made play worse. The optimism bias mostly cancels when comparing moves, so the raw values work better.
- With 64 scenarios per state, roughly 90% of scenarios are settled by assignment bounds alone. The remaining scenarios cost one HiGHS call per state, and those calls dominate run time.

Raw results are in `analysis/*.json`.

### Ideas for what's next

1. **Exact DP oracle.** Solve all ~1M scorecard states by backward induction, using the same `turn.py` widget for each state. That gives the true optimum for these rules and a per-move regret measure for the MILP agent.
2. **Bias correction.** Fit a correction for the hindsight value as a function of turns left and open categories. Alternatively, add a nonanticipativity penalty as in information-relaxation duality (Brown, Smith and Sun, 2010).
3. **Variance and speed.** Use more scenarios, a precomputed value table to replace the on-line MILP, or HiGHS warm starts.
4. **Distillation.** Use this agent, or the DP oracle, as the teacher for a small neural net, as your MLOps imitation experiment did with the heuristic. That gives instant moves for the Streamlit app.

## Run

```bash
cd Yahtzee/MonteCarlo_MILP
pip install -r requirements.txt
export PYTHONPATH=src

python -m pytest                                              # tests, incl. MILP vs brute force
python -m yahtzee_mcmilp.cli play --seed 7                    # one logged game
python -m yahtzee_mcmilp.cli simulate --games 200 --workers 4 # benchmark
python -m yahtzee_mcmilp.cli simulate --agent greedy --games 2000
python -m yahtzee_mcmilp.cli value                            # hindsight value of a new game
```

Options: `--scenarios` (Monte Carlo sample size, default 64), `--discount` (default 1.0), `--bank-seed`.

## Rules

Same as the rest of this repo so the numbers are comparable: 13 boxes, +35 upper bonus at 63, +100 for each extra Yahtzee once the Yahtzee box holds 50, and **no joker rules**. The published optimum, 254.59, includes joker rules. Without the Yahtzee bonus and joker the optimum is 245.87. The optimum for these exact rules lies between the two, so treat about 250 as the ceiling.
