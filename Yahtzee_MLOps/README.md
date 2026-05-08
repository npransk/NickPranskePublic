# Yahtzee MLOps Bot

This is a fresh rebuild of the Yahtzee bot in a separate project folder. The original `Yahtzee` folder is intentionally untouched.

## What changed

- One shared game engine for training, CLI use, tests, and Streamlit.
- Dice are represented by face counts instead of physical positions, reducing duplicate states and actions.
- Dueling Double DQN uses valid-action masks for next-state targets, so the model does not learn from impossible moves.
- Local SQLite telemetry records human and AI gameplay events.
- Local model registry saves timestamped models and updates `models/latest.pt`.
- Pipeline command checks for new gameplay events and triggers incremental training.
- Tests cover core scoring and environment behavior.

## Setup

```powershell
cd "C:\Users\prans\OneDrive\Documents\Code Projects\Python\NickPranskePublic\Yahtzee_MLOps"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Commands

Run tests:

```powershell
pytest
```

Train a fresh model:

```powershell
yahtzee-mlops train --episodes 10000 --eval-games 200 --fresh
```

Train a usable model quickly by imitating the heuristic baseline:

```powershell
yahtzee-mlops imitate --episodes 2000 --epochs 3 --eval-games 500 --hidden-size 128 --batch-size 256
```

Run a faster RL fine-tune:

```powershell
yahtzee-mlops train --episodes 5000 --eval-games 500 --hidden-size 128 --batch-size 64 --min-replay-size 500 --train-every-steps 4 --epsilon-decay 0.9997
```

Benchmark random and heuristic players:

```powershell
yahtzee-mlops benchmark --games 1000
```

Start the Streamlit app:

```powershell
streamlit run app.py
```

Run the incremental pipeline after people have played:

```powershell
yahtzee-mlops pipeline --min-new-events 100 --episodes-per-run 2000
```

Install daily overnight training:

```powershell
.\scripts\install_nightly_task.ps1 -Time "02:00" -MinNewEvents 100 -Episodes 2000 -EvalGames 100
```

See [OPERATIONS.md](OPERATIONS.md) for what the pipeline trains on, what it costs, and how model promotion should work.

Inspect telemetry:

```powershell
yahtzee-mlops telemetry
```

## How "trains as people play" works

1. The app logs gameplay events to `data/gameplay.sqlite3`.
2. The pipeline checks for unused gameplay events.
3. When enough fresh events exist, it resumes from `models/latest.pt`, trains more self-play episodes, evaluates, registers a new timestamped model, and marks those events as consumed.

The current pipeline uses gameplay telemetry as a trigger and measurement source. The next upgrade would convert human score decisions into offline imitation or preference-learning examples, then mix those with self-play replay.

## What I still need from you

- Deployment target: local-only, Streamlit Community Cloud, a VPS, or something like AWS/GCP/Azure.
- Retraining policy: train every N games, every night, or only when performance drops.
- Compute budget: CPU only, local GPU, Colab, or cloud GPU.
- Data policy: anonymous local telemetry only, named players, or cloud-stored gameplay history.
- Success metric: mean score, win rate against humans, over-200 rate, or benchmark vs heuristic bots.
- Whether you want MLflow/W&B/DVC now, or whether the local registry is enough for the next iteration.

## Suggested next upgrades

- Add a heuristic baseline player and compare every model against it.
- Use MLflow or Weights & Biases for experiment tracking.
- Add an offline learning step from logged human scoring choices.
- Schedule `yahtzee-mlops pipeline` with Windows Task Scheduler or GitHub Actions.
- Containerize the app and pipeline for deployment.
