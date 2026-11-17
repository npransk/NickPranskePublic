# 8. Review a finished game with Claude, instead of judging picks with a local model

Date: 2026-11-17 · Status: accepted (replaces the judge in [ADR 7](0007-measure-the-ai-and-judge-it-locally.md))

## Context
ADR 7 planned a local judge: an open model (Gemma 4 12B by default) on the desktop, grading each strategist pick on four pass/fail checks. It was built and tested against a stand-in server, but never run on a real model. Looking at it again before doing so:

- A 12B model reasons less well than the Sonnet 5.5 strategist it would grade. The checks that need no judgment (the pick's engine rank, the reason's length, invented hands) were already scored in code.
- It graded a pick's one-sentence reason without knowing how the game went.
- It needed the desktop, Ollama and a calibration run before its scores meant anything. With a game or two a week it would have had few picks to grade.
- It told the player nothing: its verdicts went only to Langfuse.

What Nick wants from a game is feedback on his own play as well as the strategist's.

## Decision
- **A "Score My Game" button** on a finished game. A game now also ends without a winner: a "wall game", when the tiles run out (`{"type": "wall"}`).
- **The engine replays the game and works out the facts** (`mahjong_engine.hindsight`, no AI). At each Charleston pass and each of the player's discards it records what the engine would have done, what the player did, and how far the closest hand was afterwards either way. It also lists the discards the player could have called, any Mahjong they had and didn't declare, any discard an opponent's likely hand needed, and, for each strategist pick, where it stood when asked and how that hand fared afterwards. Strategist picks are now kept with the game (`games/<id>.ai.json`), along with the player's "I'll go for it" tap.
- **Claude Opus 5.5 writes the review from those facts only.** It gives:
  - a 1-10 score for the player's play;
  - what went well;
  - 3-5 moments that mattered;
  - a good, reasonable or poor verdict on each strategist pick, judged on what could be known when it was asked;
  - one thing to work on.

  A JSON schema fixes the shape. Code drops any point that cites a move or an ask that doesn't exist, and counts the drops.
- **Opus, not Sonnet.** The reviewer is a stronger model than the strategist it grades, so the strategist isn't marking its own work. It runs once a game, so the price matters less: about $0.10-0.20 a review, estimated, not yet measured.
- **The review is kept with the game**, so reopening it costs nothing; "Score again" replaces it.
- **In Langfuse**, each review is one trace in the game's session: an evaluator observation holding the engine's replay and the Claude call. Its scores:
  - `review_play_score` on the game;
  - `strategist_hindsight` on each pick's own trace;
  - `review_outcome` and `review_dropped_refs` on the review;
  - the player's "Useful?" tap as `review_helpful`.

  These take the judge's place on the dashboards and in the drift report.
- **The judge's objects already in Langfuse are left in place, unused**: its prompt, the calibration dataset, its score configs and its dashboard tiles. `langfuse_setup.py` now keeps any tile it doesn't place on its boards, moved below its own.

## Consequences
- Strategist picks are graded only in games that get reviewed, and only once the game is over.
- The verdicts are Claude grading Claude, though with a different and stronger model. The engine's facts limit what it can claim, and `review_helpful` records whether the player agrees. There is no hand-labelled set for the reviewer yet.
- A review costs money where the judge didn't.
- The replay ranks the card again at each of the player's decisions. It took about 2 seconds for a 160-move game on the made-up sample card in a development container. The real card is larger and the Container App has half a CPU, so expect several seconds there, before the Claude call.
- Picks asked before this change are only in Langfuse, so reviews of those games cover the player's moves alone.
