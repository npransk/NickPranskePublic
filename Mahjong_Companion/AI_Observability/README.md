# AI observability

The source behind [Part 2 of the write-up](../README.md#part-2-ai-observability). These are copies from the private repository, there to be read: they import a rules engine that is not published and read their keys from the environment, so nothing here runs or holds a secret.

Each AI module opens its own trace next to its model call, so that part of the instrumentation is in [`../LLM_Orchestration/`](../LLM_Orchestration/). This folder holds what is only about observing and grading: how a trace is labelled, how answers are scored, how a finished game is reviewed, and how change over time is looked for.

## The modules

| File | What it does | Worth a look |
| --- | --- | --- |
| [`observability.py`](observability.py) | The one place tracing is set up: the Langfuse client, the labels every trace carries, how a Claude call is recorded, and the list of every score. | `_should_export()` keeps the Anthropic SDK's own spans out, which had been doubling every cost. `SCORES` declares each score with a type, and `record_score()` refuses anything not on the list. `record_claude()` flags a call that another model answered. |
| [`hindsight.py`](hindsight.py) | From the rules engine, so no AI: replays a finished game and works out the facts the review is allowed to use. | At each of your discards, `_my_discard()` compares your choice with the engine's and with how far your closest hand was afterwards either way. `_call_chance()` finds calls you could have made and Mahjong you missed. `_picks_at()` and `_watch()` follow each strategist pick through the rest of the game. `last_needed` avoids re-ranking the card where it can be proved nothing changed. |
| [`review.py`](review.py) | Score My Game: Claude Opus 5.5 reviews a finished game from the engine's facts only. You get a 1-10 score for your play, the moments that mattered, a verdict on each strategist pick and one thing to work on. | `SYSTEM` asks for decisions to be scored, not luck, and for picks to be judged on what was knowable when they were asked. `SCHEMA` fixes the answer's shape. `_checked()` drops any point citing a move or an ask that doesn't exist. `_score()` puts the play score on the game and each verdict on the strategist pick's own trace. |
| [`notes.py`](notes.py) | Keeps the strategist's picks, your "go for it" taps and the review beside the game, so a review can grade the picks and reopening it costs nothing. | `add_ask()` and `mark_followed()`. |
| [`evalset.py`](evalset.py) | The fixed test material, all on the made-up sample card: nine positions for the strategist. | `Position`: a game as a list of moves, replayed by the engine. |
| [`drift.py`](drift.py) | Compares a recent window with a baseline, feature by feature, and says what moved by more than chance. | `permutation_p()` and `adjusted()` are the statistics. `compare()` turns them into steady, worse, better, watch or not enough data. `Changes` lists new models, prompts and releases. |
| [`experiments.py`](experiments.py) | Runs the fixed positions as a Langfuse experiment, so runs can be compared side by side. | `run_strategist()`. |
| [`history.py`](history.py) | Reads observations and scores back from Langfuse, a page at a time. | Timeouts and retries: the SDK gives a read 5 seconds, and a page with inputs can take longer. |
| [`langfuse_setup.py`](langfuse_setup.py) | What the Langfuse project should contain, as code: score definitions, the prompt registry, the dataset and the dashboards. | `BOARDS` is three dashboards as data. `check_boards()` asks Langfuse each widget's question read-only before anything is created. `sync_boards()` changes only what differs, and `with_kept_tiles()` keeps tiles it doesn't place (added by hand, or the retired judge's). |
| [`selftest.py`](selftest.py) | Sends one made-up action per feature through the real code into Langfuse, reads it back and checks eleven things. | `check()` is the list of what has to be true of a healthy trace. |
| [`playbyplay.py`](playbyplay.py) | Writes every move, fix and undo to the game's Langfuse session as an event named like the move, with its source and the player's position after it. | `record_summary()` adds the "Game over" entry (after Mahjong or a wall game), scores the game as a whole and flushes, because the app may scale to zero right after. `_record()` swallows any Langfuse error so recording can never break a game. |
| [`games.py`](games.py) | The game API. It is where each AI call starts and where each tap comes back as a score. | `review_game()`, `strategist_feedback()`, `review_feedback()`, `photo_feedback()` and `heard_feedback()`. |

## The scripts that run them

| File | What it does | Cost |
| --- | --- | --- |
| [`scripts/drift.py`](scripts/drift.py) | Prints the drift report. Exits with 1 when something got worse, so a scheduled run can raise the alarm. | Free |
| [`scripts/strategist_eval.py`](scripts/strategist_eval.py) | Asks the strategist about the nine fixed positions and records the run as a Langfuse experiment. | About $0.10 |
| [`scripts/langfuse_setup.py`](scripts/langfuse_setup.py) | Brings the Langfuse project in line with the code. `--check` is a read-only dry run. | Free |
| [`scripts/langfuse_smoke.py`](scripts/langfuse_smoke.py) | The end-to-end self-test. | Free |
| [`scripts/vision_eval.py`](scripts/vision_eval.py) | Runs the photo reader over labelled single-tile photos and scores each reading. | $0.10 to $0.20 |
| [`scripts/draft_card.py`](scripts/draft_card.py) | Runs the card drafter and, with `--compare`, scores the draft against a known card. | $0.25 to $1 |

Anything that spends money refuses to run without `--yes`.

## The tests

| File | What it shows |
| --- | --- |
| [`tests/test_tracing_e2e.py`](tests/test_tracing_e2e.py) and [`tests/tracing_probe.py`](tests/tracing_probe.py) | The real Anthropic and Langfuse SDKs, talking to one local stand-in server that plays both Claude and Langfuse. With the app's filter each Claude call reaches Langfuse once. Without it, twice. |
| [`tests/test_hindsight.py`](tests/test_hindsight.py) | The engine's replay on the made-up sample card: a winning discard let go is a missed Mahjong, a discard is set against the engine's choice, a discard an opponent's likely hand needs is flagged, and a strategist pick is followed to the end of the game. |
| [`tests/test_review.py`](tests/test_review.py) | Score My Game against a scripted stand-in for Claude's streamed reply. It checks what Claude is told (the engine's facts and the strategist's words, nothing else) and that invented moves and asks are dropped. It also checks the scores and where they land, that a refusal is reported and not kept, and that reopening a game doesn't ask Claude again. |
| [`tests/test_drift.py`](tests/test_drift.py) | When a change counts as drift: large and unlikely to be chance. Small moves are steady however sure, big moves on little data are only watched, and one slow call is not drift. |
| [`tests/conftest.py`](tests/conftest.py) | The stand-in for Langfuse that the tests record into, and the rule that no test ever reaches the real one. |

## Where each trace and score is created

| In Langfuse | Type | Created in |
| --- | --- | --- |
| `strategist` | trace: agent, with a generation per Claude call and a tool observation per tool call | `strategist.py` |
| `heard` | trace: chain, with a `parse-line` span and a `jev` generation when Jev is asked | `listen.py` |
| `read-hand` | trace: one generation | `vision.py` |
| `draft-card` | trace: agent, with a generation and a `validate` tool observation per round | `card_draft.py` |
| `review-game` | trace: evaluator, with a `hindsight` span for the engine's replay and a `review` generation for Claude | `review.py` |
| "12. Player 3 discards 9 bam" and so on | event, tagged `play-by-play` | `playbyplay.py` |
| `recommendation_followed` | score from a tap, boolean | `games.py` |
| `tile_cv_correct`, `tile_fixes` | scores from a tap | `vision.py` |
| `discard_parse_correct`, `heard_outcome` | scores from a tap, with a fixed ID per line so a later correction overwrites them | `listen.py` |
| `pick_engine_rank`, `reason_words`, `reason_within_limit`, `strategist_outcome`, `strategist_llm_calls`, `strategist_tool_calls`, `strategist_rejected_picks` | scores from checks in code | `strategist.py` |
| `tiles_read`, `tiles_uncertain` | scores from checks in code | `vision.py` |
| `card_draft_accuracy`, `card_draft_rounds`, `card_draft_clean` | scores from checks in code | `card_draft.py` |
| `game_won`, `game_moves`, `discards_auto_logged`, `game_fixed_moves` | scores on the whole game (the session) | `playbyplay.py` |
| `strategist_hindsight` | score from the game review, on each strategist pick's own trace: good, reasonable or poor | `review.py` |
| `review_play_score` | score from the game review, on the whole game: 1 to 10 | `review.py` |
| `review_outcome`, `review_dropped_refs` | scores from checks in code, on the review | `review.py` |
| `review_helpful` | score from a tap on the review | `review.py` |
| `picked_closest` | score on experiment runs | `experiments.py` |

Every trace and event carries the game's ID as its session, so one game is one timeline.

The Langfuse client is always fetched through one function, `observability.langfuse_client()`. It picks project-specific key names first, which keeps this app's traces out of the other Langfuse project my laptop is set up for.
