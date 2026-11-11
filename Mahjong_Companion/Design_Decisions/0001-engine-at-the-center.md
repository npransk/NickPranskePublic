# 1. A deterministic engine sits at the center; AI sits at the edges

Date: 2026-10-23 · Status: accepted

## Context
The app has to count tiles, enumerate card hands and compute odds. LLMs are unreliable at counting, but good at reading photos, parsing messy speech and weighing soft signals.

## Decision
`packages/engine` is plain, fully tested Python with no AI. Vision, speech and agents turn raw input into engine events, or choose and explain among options the engine has already scored.

## Consequences
- Every AI component can be measured against the engine's ground truth.
- The app works with tap input alone, before vision or audio exist.
