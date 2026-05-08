# Training And Score Analysis

Generated on 2026-05-08.

## Runs

| Player / Run | Games | Mean | Median | Std Dev | Min | Max | Over 200 | Over 250 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Random baseline | 500 | 45.55 | 41.00 | 17.70 | 15 | 110 | 0.0% | 0.0% | Valid random actions |
| Heuristic baseline | 500 | 188.33 | 175.50 | 65.94 | 73 | 508 | 36.6% | 12.4% | Deterministic rule policy |
| Fast DQN | 200 | 116.19 | 113.00 | 16.89 | 82 | 225 | 0.5% | 0.0% | 1,000 episodes, small network, train every 4 steps |
| Heuristic imitation model | 500 | 180.57 | 174.00 | 63.76 | 60 | 490 | 32.4% | 10.6% | 2,000 teacher games, 102,204 examples, 3 epochs |

## Takeaways

- Pure DQN is much faster with smaller batches, a smaller network, shorter replay warmup, and less frequent gradient updates, but it still needs a lot of experience to become good.
- Heuristic imitation is the best speed improvement so far. It produced a usable neural model in about 98 seconds on this machine.
- The app now uses a quality gate: if `models/latest_metrics.json` reports a mean score below 170, the app falls back to the heuristic baseline instead of serving a weak trained model.
- The current best registered model is `models/20260508T164306Z/model.pt`.

## Recommended Training Recipe

1. Start with imitation:

```powershell
yahtzee-mlops imitate --episodes 2000 --epochs 3 --eval-games 500 --hidden-size 128 --batch-size 256
```

2. Fine-tune with RL using faster settings:

```powershell
yahtzee-mlops train --episodes 5000 --eval-games 500 --hidden-size 128 --batch-size 64 --min-replay-size 500 --train-every-steps 4 --epsilon-decay 0.9997
```

3. Compare against baselines:

```powershell
yahtzee-mlops benchmark --games 1000
```
