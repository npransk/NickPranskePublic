# 2. NMJL card data never enters git

Date: 2026-10-23 · Status: accepted

## Context
The yearly NMJL card is copyrighted. The project may later be shown publicly.

## Decision
Card data (expanded hands under a hand ID, with a plain label such as "Any Like Numbers, 2nd from bottom") and all photos live in the private `cards` and `tile-photos` blob containers. Tests use made-up sample hands.

## Consequences
The repo can be made public at any time without scrubbing history.
