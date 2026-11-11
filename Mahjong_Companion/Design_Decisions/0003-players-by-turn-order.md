# 3. Attribute discards by turn order, not by voice

Date: 2026-10-23 · Status: accepted

## Context
Telling four voices apart in a noisy room is unreliable.

## Decision
The user is always Player 1 and enters the player count at game start. A discard belongs to the player whose turn it is; a call moves the turn to the caller. Every inferred event is shown as "Player 2 discards 1 crak" for one-tap confirm or fix.

## Consequences
No speaker diarization is needed. Corrections become Langfuse scores.
