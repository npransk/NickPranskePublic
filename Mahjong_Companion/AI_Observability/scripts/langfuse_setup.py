# Showcase copy of scripts/langfuse_setup.py from the private mahjong-companion repo. Here to be read, not run.

"""Bring the Langfuse project in line with the code. Safe to run again: it changes only what
differs.

    uv run --env-file .env python scripts/langfuse_setup.py --check   # read-only dry run
    uv run --env-file .env python scripts/langfuse_setup.py

What it sets up (the definitions live in ``mahjong_api.langfuse_setup``):

- a score config for every score the app writes, so Langfuse explains each one and offers it
  for annotating traces by hand;
- each prompt's text under a ``sha-<fingerprint>`` label, so calls link to the prompt version
  that produced them. Run this after changing a prompt, or the new version's calls carry only
  the fingerprint;
- the fixed test set as a dataset (``strategist-positions``);
- three dashboards: "Mahjong: AI quality", "Mahjong: cost and tokens" and "Mahjong: judge and
  evals" (named before the judge was retired). Tiles added to them by hand are kept. The
  existing "Mahjong Companion" board is kept; its cost tiles stop double-counting and it is
  limited to real games.

``--check`` asks Langfuse each dashboard widget's question without creating anything, to catch
a query Langfuse no longer accepts. ``--only`` limits the run to some parts.
"""

from __future__ import annotations

import argparse

from mahjong_api import langfuse_setup as setup
from mahjong_api import observability

PARTS = ("scores", "prompts", "datasets", "dashboards")


def summary(done: dict[str, list[str]]) -> str:
    return ", ".join(f"{len(names)} {what}" for what, names in done.items() if names) or "nothing"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Read-only: test the widget queries")
    parser.add_argument(
        "--only", default=",".join(PARTS), help=f"Comma-separated parts: {', '.join(PARTS)}"
    )
    args = parser.parse_args()
    parts = [part for part in args.only.split(",") if part]
    if unknown := set(parts) - set(PARTS):
        raise SystemExit(f"Unknown part(s): {', '.join(sorted(unknown))}. Choose from {PARTS}.")

    langfuse = observability.langfuse_client()
    if not langfuse.auth_check():
        raise SystemExit("Langfuse keys are missing or wrong (see .env.example).")

    if args.check:
        widgets = sum(len(board.widgets()) for board in setup.BOARDS)
        print(f"Asking Langfuse the {widgets} dashboard questions (nothing is created)...")
        problems = setup.check_boards(langfuse)
        for problem in problems:
            print(f"  PROBLEM {problem}")
        print("Every widget's query is accepted." if not problems else f"{len(problems)} problems.")
        return 1 if problems else 0

    if "scores" in parts:
        done = setup.sync_score_configs(langfuse)
        print(f"Score configs: {summary(done)}")
        for name in done["mismatched"]:
            print(f"  {name} exists in Langfuse under another type: archive it there and rerun")
    if "prompts" in parts:
        done = setup.sync_prompts(langfuse)
        print(f"Prompts: {summary(done)}")
        for name in done["created"]:
            print(f"  registered {name}")
    if "datasets" in parts:
        for name, count in setup.sync_datasets(langfuse).items():
            print(f"Dataset {name}: {count} items in step with the code")
    if "dashboards" in parts:
        done = setup.sync_boards(langfuse)
        print(f"Dashboards and widgets: {summary(done)}")
        for line in setup.correct_overview(langfuse):
            print(f"  corrected: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
