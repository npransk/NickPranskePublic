# 7. Measure every AI step the same way, and judge it with a local model

Date: 2026-11-16 · Status: accepted; the local judge was replaced by the game review on 2026-11-17 ([ADR 8](0008-review-a-finished-game-with-claude.md)), the rest stands

## Context
Every AI step already sent a trace to Langfuse, and the player's taps became three scores. That was not enough to answer "is the AI still as good, and as cheap, as it was?":

- Traces could not be compared. Eval runs and real games shared one environment, and no trace said which release or which version of a prompt produced it.
- Costs were counted twice. The Anthropic SDK emits its own OpenTelemetry span for each API call, and Langfuse stored it as a second generation next to the app's.
- A tap says whether you followed a pick, not whether its reason was true. Nothing checked the strategist's one sentence against the engine's numbers.
- A model behind a fixed name can change, and so can the games. A dip in quality could be either, and nothing told them apart.

Langfuse can run an LLM judge itself, but that needs a hosted model and pays per verdict. The Container App (0.5 CPU, scales to zero) cannot hold a model.

## Decision
- **One tracing module** (`mahjong_api.observability`). Every trace carries its environment (`production`, `development`, `eval`, `judge`, `selftest`), the release (the image's git commit), a feature tag, the game as its session, a hash of the signed-in player and a fingerprint of the prompt. The Anthropic SDK's own spans are not exported.
- **Three kinds of score, declared in one list** (`observability.SCORES`) with a type each:
  - *taps*: what the player did (followed the pick, fixed tiles, fixed a heard line);
  - *code checks*: what can be counted without judgment (where the pick stood in the engine's ranking, words in the reason, calls per ask, how the ask ended, whole-game scores on the session);
  - *the judge*: four pass/fail verdicts on each strategist pick.
- **The judge is an open model on the desktop**, served by Ollama (`scripts/judge.py`). It reads picks back from Langfuse and writes its verdicts onto their traces. It is a different model family from the one it judges, answers pass or fail on one thing at a time, and writes its evidence before each verdict. Counting is done for it in code.
- **The judge is measured before it is trusted.** `judge.py calibrate` runs it on hand-labelled answers to made-up positions, several deliberately broken, and reports agreement, Cohen's kappa and false passes per check. Every verdict carries the judge model and a fingerprint of the rubric.
- **Drift is looked for in two ways.** `scripts/drift.py` compares a recent window with a baseline for every token, cost, latency and quality metric, and calls a change drift only when it is large and a permutation test (corrected for the number of metrics) says it is unlikely to be chance. `scripts/strategist_eval.py` asks the strategist about nine positions that never change and records each run as a Langfuse experiment, so a difference between runs is the model or the prompt, not the games.
- **The Langfuse project's setup is code** (`scripts/langfuse_setup.py`): score configs, prompts mirrored under a `sha-<fingerprint>` label, the two datasets and the dashboards. `scripts/langfuse_smoke.py` checks the wiring end to end in an environment of its own (`selftest`) that nothing else reads.

## Consequences
- Judge scores arrive after a game, when the judge is run on the desktop, not during play.
- The judge is only as good as its calibration. Eighteen labelled answers are a start; real picks where the judge and the player disagree should be added.
- With a game or two a week most drift metrics need weeks before they have enough data. The fixed positions answer sooner, for about $0.10 a run.
- A changed prompt must be registered again (`langfuse_setup.py`), or its calls carry the fingerprint without the link.
- A strategist ask now writes eight scores instead of one, which counts toward Langfuse's usage.
- Traces from before this decision sit in Langfuse's `default` environment, eval runs included. Dashboards and reports read `production` and `default` together and leave out the old eval session where Langfuse allows it.
- OpenTelemetry's GenAI attribute names are still marked as in development, so the app records model calls through Langfuse's own observation types and does not depend on them.
