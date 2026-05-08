# Yahtzee Expectimax

This is a third, separate rebuild aimed at high scores rather than learning infrastructure. It does not edit the original `Yahtzee` folder or the `Yahtzee_MLOps` folder.

The strategy is expectimax dynamic programming:

- Represent dice as face counts, not physical dice positions.
- Enumerate legal keeps after each roll.
- Compute the expected value of every keep.
- Compute the best score category for every scorecard state.
- Cache recursive values so repeated states are cheap.

This is closer to mathematical Yahtzee play than a neural bot. It is the right direction if the target is an average score around 240+.

## Run

```powershell
cd "C:\Users\prans\OneDrive\Documents\Code Projects\Python\NickPranskePublic\Yahtzee_Expectimax"
$env:PYTHONPATH = "src"
python -m yahtzee_expectimax.cli ev
python -m yahtzee_expectimax.cli simulate --games 100 --output analysis\simulation_100.json
python -m yahtzee_expectimax.cli tournament --games 200 --output analysis\tournament_200.json
python -m yahtzee_expectimax.cli turbo --games 10000 --output analysis\turbo_10000.json
python -m yahtzee_expectimax.cli search --iterations 60 --games 2000 --output analysis\numba_search.json
python -m yahtzee_expectimax.cli validate --params-file analysis\numba_refine_40x3000.json --games 10000
```

## Notes

The solver includes upper-section bonus and Yahtzee bonus handling. Joker rules are simplified, which keeps the state space much smaller while still preserving the main source of high-scoring Yahtzee play.

See `analysis/RESULTS.md` for the current benchmark results. The best validated policy is around 236, while smaller tuning samples can cross 240 but do not hold up under larger independent validation. The next real jump requires full scorecard-state dynamic programming rather than turn-local utility tuning.
