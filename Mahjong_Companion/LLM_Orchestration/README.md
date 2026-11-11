# LLM orchestration

The source behind [Part 1 of the write-up](../README.md#part-1-llm-orchestration). These are copies from the private repository, there to be read. They import a rules engine that is not published, and they read their keys from the environment, so nothing here runs or holds a secret.

| File | What it does | Worth a look |
| --- | --- | --- |
| [`strategist.py`](strategist.py) | The agent that picks which hand to go for. Claude Sonnet 5.5 at low effort, three strict tools, at most 4 model calls. | `ask()` is the whole loop. `TOOLS` and `SYSTEM` are the contract with the model. |
| [`test_strategist.py`](test_strategist.py) | Tests for the loop with a scripted stand-in for Claude: no key, no network, no cost. | A made-up hand is rejected, a text answer is nudged toward the tool, the loop gives up after the call limit, a refusal is an error. |
| [`vision.py`](vision.py) | Reads a photo of a rack into tile names. One call with a JSON schema that only allows the engine's 36 tile names. | `PROMPT` describes the markings of the actual tile set, which is what took accuracy from 40 to 42 of 43 photos. |
| [`card_draft.py`](card_draft.py) | Drafts a year's card from a photo. Claude Opus 5.5 at high effort writes it, the engine's validator checks it, problems go back for up to 4 rounds. | `draft_card()` is the repair loop. `compare()` scores a draft by meaning, not spelling. |
| [`sample_card.toml`](sample_card.toml) | A made-up card in the engine's format. It is the example the drafter is shown, and the card behind every screenshot. | Not from any real card. |
| [`listen.py`](listen.py) | Turns one heard line into a proposed game event: grammar parser first, Jev only when the parser is unsure, the player decided by turn order. | `hear()` and `_ask_jev()`. `score()` writes the player's answer back as a Langfuse score. |
| [`jev.py`](jev.py) | The client for Jev (TypeSafe): two typed questions per line, a yes/no and a choice over 36 tiles, with a measured cut-off. | The comment above `GAME_TALK` is the calibration data. |

## Which model does which job

| Job | Model | Why |
| --- | --- | --- |
| Pick a hand | Claude Sonnet 5.5, low effort | A quick judgment over numbers the engine has already computed. I may ask several times a game, so it has to be cheap. |
| Read a photo | Claude Sonnet 5.5 vision, low effort | Reading printed indexes off tiles is not hard reasoning, and the schema catches anything that isn't a tile. |
| Draft a card | Claude Opus 5.5, high effort | Runs once a year, and one wrong hand misleads every game after it. |
| Filter table talk | Jev, pinned version | A yes/no and a multiple choice. No text to generate, so a typed model at about five cents per thousand lines fits better than a chat model. |
