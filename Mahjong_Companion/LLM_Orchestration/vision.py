# Showcase copy of services/api/src/mahjong_api/vision.py from the private mahjong-companion repo. Here to be read, not run.

"""Read a hand of tiles from a photo with Claude's vision (Phase 5, the vision-LLM baseline).

Claude answers in a fixed JSON shape whose tile names are limited to the 36 the engine knows, so
the reading drops straight into the tile picker. Tiles it isn't sure of are flagged for the
player to check; every fix the player makes becomes the ``tile_cv_correct`` and ``tile_fixes``
scores in Langfuse, which are the baseline a trained detector will be measured against later.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from mahjong_engine.tiles import FULL_SET

from mahjong_api.observability import (
    flush,
    langfuse_client,
    marking_errors,
    prompt_link,
    prompt_version,
    record_claude,
    record_score,
    trace,
)

MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MAX_TOKENS = 4096
PROMPT_NAME = "read-hand"
MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
TILE_NAMES = [str(t) for t in sorted(FULL_SET)]

PROMPT = """This photo shows one player's American Mahjong tiles, usually standing in a rack. \
List every tile you can see, left to right (top row first if there are two rows).

This set marks every tile with a red index in a corner (it may be printed sideways, so an N \
can look like a Z). Read the index first, then check it against the picture:
- Bams: a red number; green bamboo drawings. The 1 bam is a bird; the 8 bam looks like a \
green W above an M.
- Craks: a red number; a black Chinese numeral beside a red 萬.
- Dots: a red number; coloured circles.
- Dragons: "C" is the red dragon (red 中); "F" is the green dragon (green 發), not a flower; \
"P" is the white dragon, called soap (an empty double frame).
- Winds: "N", "E", "W" or "S" with a black Chinese character (北, 東, 西, 南).
- Flowers: either a season index ("SPR", "SUM", "AUT", "WIN") or a red number next to a black \
Chinese character such as 竹 or 梅, with a coloured plant or flower picture. Even when the \
picture is bamboo, a tile with a black character beside its red number is a flower, not a bam. \
All flowers count as one tile, "flower".
- Jokers: "JOKER" down the side.

Mark a tile uncertain when you could be wrong about it. Don't guess tiles you can't see."""

SCHEMA = {
    "type": "object",
    "properties": {
        "tiles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tile": {"type": "string", "enum": TILE_NAMES},
                    "uncertain": {"type": "boolean"},
                },
                "required": ["tile", "uncertain"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tiles"],
    "additionalProperties": False,
}


# Changes whenever the instructions or the answer shape do; on every photo-reading trace.
PROMPT_VERSION = prompt_version(PROMPT, SCHEMA)


class VisionError(RuntimeError):
    """The photo couldn't be read."""


@dataclass(frozen=True)
class Reading:
    tiles: list[str]  # left to right
    uncertain: list[int]  # indexes into ``tiles``
    trace_id: str | None


def read_hand(client: Any, image_b64: str, media_type: str, game_id: str) -> Reading:
    """Read the tiles in a base64 photo. ``client`` is an ``anthropic.Anthropic``."""
    if media_type not in MEDIA_TYPES:
        raise VisionError(f"Unsupported image type {media_type}")
    if len(image_b64) * 3 // 4 > MAX_IMAGE_BYTES:
        raise VisionError("The photo is too large; the app should shrink it first")
    langfuse = langfuse_client()
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": image_b64},
    }
    image_bytes = len(image_b64) * 3 // 4
    try:
        with (
            trace(
                "read-hand",
                game_id=game_id,
                feature="vision",
                version=PROMPT_VERSION,
                prompt=prompt_link(PROMPT_NAME, PROMPT_VERSION),
                image_bytes=image_bytes,
            ),
            langfuse.start_as_current_observation(
                name="read-hand",
                as_type="generation",
                model=MODEL,
                model_parameters={"effort": EFFORT, "max_tokens": MAX_TOKENS},
                # The photo itself is left out of the trace; its size is enough to compare runs.
                input={"prompt": PROMPT, "image_bytes": image_bytes, "media_type": media_type},
            ) as generation,
            marking_errors(generation),
        ):
            trace_id = langfuse.get_current_trace_id()
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": [image, {"type": "text", "text": PROMPT}]}],
                output_config={
                    "effort": EFFORT,
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            # Record the call before judging its answer: a reading that fails still cost tokens.
            record_claude(generation, response, MODEL, text)
            if response.stop_reason == "refusal":
                raise VisionError("The model declined to read the photo")
            try:
                found = json.loads(text)["tiles"]
            except (ValueError, KeyError) as e:
                raise VisionError("The model's answer couldn't be read") from e
            generation.update(output=found)
        tiles = [t["tile"] for t in found]
        uncertain = [i for i, t in enumerate(found) if t["uncertain"]]
        for name, value in (("tiles_read", len(tiles)), ("tiles_uncertain", len(uncertain))):
            record_score(langfuse, name, value, trace_id=trace_id, score_id=f"{trace_id}-{name}")
    finally:
        flush()
    return Reading(tiles, uncertain, trace_id)


def fixes_needed(read: list[str], final: list[str]) -> int:
    """How many tiles the player had to change, add or remove to get from ``read`` to ``final``."""
    read_counts, final_counts = Counter(read), Counter(final)
    return max((read_counts - final_counts).total(), (final_counts - read_counts).total())


def score_reading(trace_id: str, read: list[str], final: list[str]) -> float:
    """Record how good a reading was: the share of the final tiles it got right, and how many
    tiles the player had to fix. One of each per reading; a second answer overwrites."""
    fixes = fixes_needed(read, final)
    correct = max(0.0, 1 - fixes / max(len(final), 1))
    langfuse = langfuse_client()
    comment = f"{fixes} fix{'es' if fixes != 1 else ''} to {len(final)} tiles"
    for name, value in (("tile_cv_correct", correct), ("tile_fixes", fixes)):
        record_score(
            langfuse,
            name,
            value,
            trace_id=trace_id,
            score_id=f"{trace_id}-{name}",
            comment=comment,
        )
    flush()
    return correct
