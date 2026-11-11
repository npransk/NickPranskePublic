# 5. Cards are TOML files in a notation that mirrors the printed card

Date: 2026-10-23 · Status: accepted

## Context
The card encodes suits by ink color, offers "-or-" alternatives, and uses "Any" notes for numbers that can change. The app needs every concrete 14-tile target behind each line, grouped under one ID that a player can find on the physical card.

## Decision
- One TOML file per year, sections in card order. A hand ID is `<year>-<section code>-<line>`, for example `2024-ALN-2`, with a label such as "Any Like Numbers, 2nd from top".
- Patterns copy the card's text. `:g`, `:r` and `:b` mark the green, red and blue ink, and different letters always mean different suits. Lowercase letters are number or wind variables, so "Any Like Numbers" and "Any 3 Consec. Nos." are `vars` instead of hand-written variants.
- `mahjong_engine.card` expands each hand into every variant and validates it: 14 tiles, no more of a tile than the set holds unless jokers can cover it (sets of 3 or more only).
- Check a card with `uv run python -m mahjong_engine.card <file>`.

## Consequences
- The 2024 card is 55 hands and 1,683 variants.
- The format is plain enough for a vision LLM to draft from a photo, with the validator as the safety net.
