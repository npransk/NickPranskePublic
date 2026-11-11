# Showcase copy of services/api/src/mahjong_api/card_draft.py from the private mahjong-companion repo. Here to be read, not run.
# One example hand in the prompt below was swapped for a made-up one, so no line of a real card appears here.

"""Draft a year's card file from a photo of the printed card (Phase 5, once a year).

Claude reads the card panel by panel and writes it in the card format (ADR 5). The engine's own
parser and validator check each draft; any problem goes back to Claude to fix, for a few rounds.
The player still reviews the draft once a year (ADR 2: card data stays private, so drafts are
written under ``data/private/cards/drafts/`` and never committed).

``compare`` scores a draft against a known card hand by hand, by meaning rather than spelling:
a hand counts as right when it expands to exactly the same 14-tile variants, with the same value
and concealed flag.
"""

from __future__ import annotations

import base64
import io
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langfuse import propagate_attributes
from mahjong_engine import card as card_format
from mahjong_engine.card import Card, CardError, Hand, card_from_dict, validate_card

from mahjong_api.strategist import langfuse_client

MODEL = "claude-opus-5-5"  # once a year and accuracy matters: the most capable Opus
EFFORT = "high"
MAX_ROUNDS = 4
LONG_EDGE = 1568
SAMPLE_CARD = Path(__file__).parents[4] / "packages" / "engine" / "tests" / "sample_card.toml"

SYSTEM = f"""You transcribe the yearly NMJL American Mahjong card from photos into a TOML card \
file. The photos show the card's panels left to right; they overlap a little, so ignore \
partly cut-off text at their edges. Copy every hand exactly as printed.

The card format, from the engine that will parse your file:

{card_format.__doc__}

Rules for this transcription:
- Ink colours: green -> :g, red -> :r, dark blue or black -> :b. Different colours are different \
suits. Groups with no number or dragon (F, winds) need no colour.
- One [[section]] per heading, in card order; give each a short upper-case code (e.g. the \
initials of its heading) and keep the heading as its name.
- Each line is one hand: put its printed note "(Any 2 Suits)" in note without brackets, its \
value in value, and concealed = true when the line ends in C (X means exposed).
- When a note says numbers can vary ("Any Like Nos.", "Any 3 Consec. Nos.", "Any Wind"), use \
lowercase variables in vars instead of the printed numbers, and distinct where numbers must \
differ. Keep fixed numbers ("These Nos. Only") as printed.
- Groups printed with the same number stay equal: "555 DDDD 555 DDDD (Pungs 1, 5 or 7)" is \
one variable used twice (both pungs 1s, or both 5s, or both 7s), not two independent ones. \
Use separate variables only for groups printed with different numbers, or when the note says \
the numbers differ.
- 0 is the white dragon (soap), used as zero in year hands.
- Write only the TOML, in one ```toml block, nothing else.

A made-up example of the format (not from any real card):

```toml
{SAMPLE_CARD.read_text(encoding="utf-8")}```"""


class DraftError(RuntimeError):
    """No usable draft came back."""


@dataclass
class Draft:
    toml: str
    card: Card | None
    problems: list[str]
    rounds: int
    trace_id: str | None


@dataclass
class Comparison:
    matched: list[str] = field(default_factory=list)  # truth hand IDs drafted correctly
    missed: list[str] = field(default_factory=list)  # truth hand IDs with no matching draft hand

    @property
    def accuracy(self) -> float:
        total = len(self.matched) + len(self.missed)
        return len(self.matched) / total if total else 0.0


def card_panels(photo: Path, panels: int = 3) -> list[str]:
    """Find the card in a photo and cut it into panels, as base64 JPEGs at full detail."""
    from PIL import Image

    image = Image.open(photo).convert("RGB")
    top, bottom, left, right = _card_bounds(image)
    width = (right - left) / panels
    overlap = int(width * 0.03)
    crops = []
    for i in range(panels):
        x0 = max(left, int(left + i * width) - overlap)
        x1 = min(right, int(left + (i + 1) * width) + overlap)
        crop = image.crop((x0, top, x1, bottom))
        crop.thumbnail((LONG_EDGE, LONG_EDGE))
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=90)
        crops.append(base64.b64encode(buffer.getvalue()).decode())
    return crops


def _card_bounds(image) -> tuple[int, int, int, int]:
    """The bright card on a darker table: rows and columns that are mostly white."""
    small = image.convert("L").resize((400, max(1, 400 * image.height // image.width)))
    pixels = small.load()
    w, h = small.size

    def bright(values) -> float:
        values = list(values)
        return sum(v > 180 for v in values) / len(values)

    rows = [y for y in range(h) if bright(pixels[x, y] for x in range(w)) > 0.5]
    if not rows:
        return 0, image.height, 0, image.width
    band = _longest_run(rows)
    # Within the card's rows everything bright is card; the folds between panels are darker,
    # so take first to last bright column rather than one run.
    cols = [x for x in range(w) if bright(pixels[x, y] for y in band) > 0.5]
    span = [cols[0], cols[-1]] if cols else [0, w - 1]
    scale = image.width / w
    pad = int(4 * scale)
    return (
        max(0, int(band[0] * scale) - pad),
        min(image.height, int((band[-1] + 1) * scale) + pad),
        max(0, int(span[0] * scale) - pad),
        min(image.width, int((span[-1] + 1) * scale) + pad),
    )


def _longest_run(values: list[int]) -> list[int]:
    runs, current = [], [values[0]]
    for v in values[1:]:
        if v == current[-1] + 1:
            current.append(v)
        else:
            runs.append(current)
            current = [v]
    runs.append(current)
    return max(runs, key=len)


def check(toml_text: str) -> tuple[Card | None, list[str]]:
    """Parse and validate a draft. Problems are written for Claude to fix."""
    try:
        card = card_from_dict(tomllib.loads(toml_text))
    except tomllib.TOMLDecodeError as e:
        return None, [f"Not valid TOML: {e}"]
    except (CardError, KeyError, ValueError, TypeError) as e:
        return None, [f"Doesn't fit the card format: {e}"]
    return card, validate_card(card)


def extract_toml(text: str) -> str:
    match = re.search(r"```toml\s*\n(.*?)```", text, re.S)
    return match.group(1) if match else text


def draft_card(client: Any, panels: list[str], year: int, model: str = MODEL) -> Draft:
    """Ask Claude for the card, feeding validator problems back until it is clean."""
    langfuse = langfuse_client()
    content: list[dict[str, Any]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": p}}
        for p in panels
    ]
    content.append({"type": "text", "text": f"Transcribe the {year} card. Set year = {year}."})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

    with propagate_attributes(session_id=f"card-{year}", trace_name="draft-card"):
        with langfuse.start_as_current_observation(
            name="draft-card", as_type="agent", input={"year": year, "panels": len(panels)}
        ) as agent:
            trace_id = langfuse.get_current_trace_id()
            toml_text, card, problems = "", None, ["No draft yet"]
            for round_number in range(1, MAX_ROUNDS + 1):
                response = _call(client, langfuse, model, messages, round_number)
                text = next((b.text for b in response.content if b.type == "text"), "")
                toml_text = extract_toml(text)
                card, problems = check(toml_text)
                with langfuse.start_as_current_observation(
                    name="validate", as_type="tool", input={"round": round_number}
                ) as span:
                    span.update(
                        output={"hands": len(card.hands) if card else 0, "problems": problems}
                    )
                if not problems:
                    break
                messages.append({"role": "assistant", "content": response.content})
                listed = "\n".join(f"- {p}" for p in problems[:40])
                messages.append(
                    {
                        "role": "user",
                        "content": "The validator found these problems. Check them against the "
                        f"photos and send the whole corrected file:\n{listed}",
                    }
                )
            agent.update(
                output={"rounds": round_number, "hands": len(card.hands) if card else 0},
                level="DEFAULT" if not problems else "WARNING",
            )
    langfuse.flush()
    if card is None and not toml_text.strip():
        raise DraftError("No card came back")
    return Draft(toml_text, card, problems, round_number, trace_id)


def _call(client: Any, langfuse: Any, model: str, messages: list, round_number: int) -> Any:
    with langfuse.start_as_current_observation(
        name=f"claude-round-{round_number}",
        as_type="generation",
        model=model,
        model_parameters={"effort": EFFORT},
        input={"round": round_number, "messages": len(messages)},
    ) as generation:
        # A whole card is a long answer: stream it so the request can't time out.
        with client.beta.messages.stream(
            model=model,
            max_tokens=32000,
            system=SYSTEM,
            messages=messages,
            output_config={"effort": EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
        if response.stop_reason == "refusal":
            generation.update(level="ERROR", status_message="declined")
            raise DraftError("The model declined to transcribe the card")
        usage = response.usage
        generation.update(
            output={"stop_reason": response.stop_reason},
            usage_details={
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            },
        )
        return response


def compare(draft: Card, truth: Card) -> Comparison:
    """Which hands of a known card the draft got exactly right, by meaning."""
    drafted = [_meaning(h) for h in draft.hands]
    result = Comparison()
    for hand in truth.hands:
        (result.matched if _meaning(hand) in drafted else result.missed).append(hand.id)
    return result


def _meaning(hand: Hand) -> tuple:
    try:
        variants = frozenset(tuple(sorted(v.sets)) for v in hand.variants())
    except CardError:
        variants = frozenset()
    return variants, hand.value, hand.concealed


def score(trace_id: str, comparison: Comparison) -> None:
    langfuse = langfuse_client()
    total = len(comparison.matched) + len(comparison.missed)
    langfuse.create_score(
        name="card_draft_accuracy",
        value=comparison.accuracy,
        data_type="NUMERIC",
        trace_id=trace_id,
        comment=f"{len(comparison.matched)} of {total} hands",
    )
    langfuse.flush()
