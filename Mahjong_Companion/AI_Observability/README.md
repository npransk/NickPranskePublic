# AI observability

The source behind [Part 2 of the write-up](../README.md#part-2-ai-observability). Tracing is not a separate layer in this app: each AI module opens its own trace, so most of the instrumentation lives next to the calls in [`../LLM_Orchestration/`](../LLM_Orchestration/). This folder holds the parts that are only about observing.

| File | What it does | Worth a look |
| --- | --- | --- |
| [`playbyplay.py`](playbyplay.py) | Writes every move, fix and undo to the game's Langfuse session as an event named like the move, with its source and the player's position after it. | `record_summary()` adds the "Game over" entry and flushes, because the app may scale to zero right after. `_record()` swallows any Langfuse error so recording can never break a game. |
| [`games.py`](games.py) | The game API. It is where each AI call starts and where each tap comes back as a score. | `strategist_feedback()` (`recommendation_followed`), `photo_feedback()` (`tile_cv_correct`, plus the label saved next to the photo), `heard_feedback()` (`discard_parse_correct`). |
| [`vision_eval.py`](vision_eval.py) | Offline eval: runs the photo reader over labelled single-tile photos and scores each reading. | It writes to Langfuse the same way the live app does, so an experiment and real play share one view. |
| [`draft_card.py`](draft_card.py) | Runs the card drafter and, with `--compare`, scores the draft against a known card. | Refuses to run without `--yes`, because it spends money. |

## Where each trace and score is created

| In Langfuse | Type | Created in |
| --- | --- | --- |
| `strategist` | trace: agent, with a generation per Claude call and a tool span per tool call | `strategist.py` |
| `heard` | trace: chain, with a `parse-line` span and a `jev` generation when Jev is asked | `listen.py` |
| `read-hand` | trace: one generation | `vision.py` |
| `draft-card` | trace: agent, with a generation and a `validate` tool span per round | `card_draft.py` |
| "12. Player 3 discards 9 bam" and so on | event, tagged `play-by-play` | `playbyplay.py` |
| `recommendation_followed` | score, boolean | `games.py` |
| `tile_cv_correct` | score, 0 to 1 | `vision.py` |
| `discard_parse_correct` | score, 0 or 1, with a fixed ID per line so a later correction overwrites it | `listen.py` |
| `card_draft_accuracy` | score, 0 to 1 | `card_draft.py` |

Every trace and event carries the game's ID as its session, so one game is one timeline.

The Langfuse client is always fetched through one function, `strategist.langfuse_client()`. It picks project-specific key names first, which keeps this app's traces out of the other Langfuse project my laptop is set up for.
