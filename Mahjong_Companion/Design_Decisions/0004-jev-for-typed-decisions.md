# 4. Use Jev for the game-talk filter, Claude for the hand pick

Date: 2026-10-23 · Status: accepted

## Context
[Jev](https://docs.typesafe.ai/introduction) (TypeSafe) answers typed questions about a state with a choice or score plus a confidence, and generates no text. Its docs advise splitting any judgment that weighs several factors into separate questions.

## Decision
- Clean announcements ("3 bam", "soap", "north wind") are matched by plain code. Jev handles the transcript lines that code can't match: "is this game talk?" and "which tile?". Low confidence routes to the confirm/fix card.
- Jev is not used for the hand pick. Picking a target weighs tiles needed, live tiles and opponent reads, and Jev can't explain its choice. The engine ranks the hands; Claude picks one and writes the reason.
- Jev calls are wrapped in Langfuse spans by hand, since TypeSafe documents no Langfuse integration.

## Consequences
Jev arrives with live audio (the last phase). The strategist is one Claude agent rather than an A/B test.
