# Operating The Yahtzee MLOps Loop

## Daily Overnight Training

The local operational flow is:

1. People play in the Streamlit app.
2. The app records gameplay events in `data/gameplay.sqlite3`.
3. Windows Task Scheduler runs `scripts/run_local_pipeline.ps1` overnight.
4. The pipeline checks whether enough fresh events exist.
5. If the threshold is met, it resumes from `models/latest.pt`, trains, evaluates, saves a timestamped model, and updates `models/latest.pt`.
6. Logs are written to `logs/nightly_pipeline_*.log`.

Install the scheduled task:

```powershell
cd "C:\Users\prans\OneDrive\Documents\Code Projects\Python\NickPranskePublic\Yahtzee_MLOps"
.\scripts\install_nightly_task.ps1 -Time "02:00" -MinNewEvents 100 -Episodes 2000 -EvalGames 100
```

Run the same job manually:

```powershell
.\scripts\run_local_pipeline.ps1 -MinNewEvents 100 -Episodes 2000 -EvalGames 100
```

View or edit it in Windows Task Scheduler under:

```text
Task Scheduler Library > Yahtzee MLOps Nightly Training
```

## What It Trains On Today

The current nightly pipeline trains with reinforcement learning self-play. Human gameplay is currently used as:

- A signal that new activity happened.
- Telemetry for monitoring app usage and human scores.
- A future dataset for imitation/preference learning.

In plain English: right now, once enough people have played, the bot wakes up overnight and plays simulated Yahtzee games against the environment to improve itself. It does not yet directly learn "Nick chose Full House here, copy that choice" from human moves.

## Next Step For Learning From People

To make the model train directly from people, add an offline learning phase that converts logged human decisions into training examples:

- State before the human action.
- Legal action mask.
- Human action label.
- Final game score or outcome.

Then the nightly job can run:

1. Imitation training on human decisions.
2. RL self-play fine-tuning.
3. Evaluation against random and heuristic baselines.
4. Promotion only if the new model beats the current model.

## Cost

Local training does not cost OpenAI credits or API credits. It uses your machine's CPU/GPU and electricity.

You would only pay money/credits if you choose to run training on a paid service such as:

- Google Colab paid compute.
- AWS/GCP/Azure GPU instances.
- Weights & Biases paid tiers for large experiment tracking.
- A hosted database or server.

This project does not call OpenAI APIs for training.

## Recommended Production Gates

Do not automatically promote every overnight model. Promote only if it passes gates such as:

- Mean score is at least the current production model.
- Over-200 rate does not regress.
- Evaluation uses at least 500 games.
- Model beats the heuristic baseline or clears a chosen score threshold.

The app currently avoids serving weak models by falling back to the heuristic policy when `models/latest_metrics.json` reports a mean score below 170.
