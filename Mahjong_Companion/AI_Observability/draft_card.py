# Showcase copy of scripts/draft_card.py from the private mahjong-companion repo. Here to be read, not run.

"""Draft a year's card file from a photo of the printed card, for you to review.

    uv run python scripts/draft_card.py Year_Cards/2024hands.JPG --year 2024 --yes
    uv run python scripts/draft_card.py Year_Cards/2024hands.JPG --year 2024 \
        --compare data/private/cards/2024.toml --yes

The draft goes to data/private/cards/drafts/<year>.toml (gitignored: card data stays private).
With --compare, every hand of a known card is checked against the draft by meaning, and the
accuracy is printed and scored in Langfuse as card_draft_accuracy.

This calls the Claude API: Claude Opus 5.5 at high effort costs roughly $0.25-1 per card,
depending on how many correction rounds it needs. It refuses to run without --yes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anthropic
from mahjong_api import card_draft
from mahjong_engine.card import load_card

DRAFTS = Path(__file__).parents[1] / "data" / "private" / "cards" / "drafts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photo", type=Path, help="Photo of the whole card, panels side by side")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--compare", type=Path, help="A known card file to score the draft against")
    parser.add_argument("--panels", type=int, default=3, help="Panels side by side on the card")
    parser.add_argument("--model", default=card_draft.MODEL)
    parser.add_argument("--yes", action="store_true", help="Agree to the API cost and run")
    args = parser.parse_args()
    if not args.yes:
        print(f"This calls {args.model} (roughly $0.25-1 per card). Add --yes to run.")
        return 1

    panels = card_draft.card_panels(args.photo, args.panels)
    draft = card_draft.draft_card(anthropic.Anthropic(), panels, args.year, args.model)
    DRAFTS.mkdir(parents=True, exist_ok=True)
    out = DRAFTS / f"{args.year}.toml"
    out.write_text(draft.toml, encoding="utf-8")
    hands = len(draft.card.hands) if draft.card else 0
    print(
        f"Wrote {out} after {draft.rounds} round(s): {hands} hands, {len(draft.problems)} problems"
    )
    for problem in draft.problems:
        print(f"  {problem}")

    if args.compare and draft.card:
        result = card_draft.compare(draft.card, load_card(args.compare))
        total = len(result.matched) + len(result.missed)
        print(f"\n{len(result.matched)} of {total} hands match {args.compare.name} exactly.")
        for hand_id in result.missed:
            print(f"  missed {hand_id}")
        if draft.trace_id:
            card_draft.score(draft.trace_id, result)
    return 0 if not draft.problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
