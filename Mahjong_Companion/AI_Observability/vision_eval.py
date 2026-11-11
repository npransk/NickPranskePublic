# Showcase copy of scripts/vision_eval.py from the private mahjong-companion repo. Here to be read, not run.

"""Measure the photo reader on your single-tile photos in Tiles/ (one tile per photo).

Each file name says which tile it is (1bam.JPG, northwind.JPG, sprflower.JPG, ...), so every
reading can be checked. Results go to Langfuse like the app's readings, with the
``tile_cv_correct`` score, and a summary prints here.

This calls the Claude API once per photo: 43 photos cost roughly $0.10-0.20. It refuses to run
without --yes.

    uv run python scripts/vision_eval.py --yes
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

import anthropic
from mahjong_api import vision
from PIL import Image

TILES = Path(__file__).parents[1] / "Tiles"
LONG_EDGE = 1568


def expected_tile(path: Path) -> str:
    name = path.stem.lower()
    if m := re.fullmatch(r"([1-9])(bam|crak|dot)", name):
        return f"{m[1]} {m[2]}"
    if name.endswith("flower"):
        return "flower"
    if m := re.fullmatch(r"(green|red|white)dragon.*", name):
        return f"{m[1]} dragon"
    if m := re.fullmatch(r"(north|east|west|south)wind", name):
        return f"{m[1]} wind"
    if name == "joker":
        return "joker"
    raise ValueError(f"Can't tell which tile {path.name} is")


def shrink(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((LONG_EDGE, LONG_EDGE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="Agree to the API cost and run")
    args = parser.parse_args()
    photos = sorted(p for p in TILES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if not args.yes:
        print(f"{len(photos)} photos in {TILES}. This calls Claude once per photo; add --yes.")
        return 1

    client = anthropic.Anthropic()
    right, wrong = 0, []
    for path in photos:
        expected = expected_tile(path)
        reading = vision.read_hand(client, shrink(path), "image/jpeg", "vision-eval")
        if reading.trace_id:
            vision.score_reading(reading.trace_id, reading.tiles, [expected])
        if reading.tiles == [expected]:
            right += 1
        else:
            wrong.append(f"{path.name}: expected {expected}, read {reading.tiles}")
        print(f"{'ok  ' if reading.tiles == [expected] else 'MISS'} {path.name}: {reading.tiles}")

    print(f"\n{right} of {len(photos)} read exactly right.")
    for line in wrong:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
