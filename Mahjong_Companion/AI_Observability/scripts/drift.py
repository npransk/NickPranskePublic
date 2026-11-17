# Showcase copy of scripts/drift.py from the private mahjong-companion repo. Here to be read, not run.

"""Has the AI changed? Compare the last two weeks with the two months before.

Reads what the app recorded in Langfuse and prints, feature by feature, whether tokens, cost,
latency or any quality score moved by more than chance would explain, after listing any new
model, prompt version or release (the usual cause). Nothing is written to Langfuse.

    uv run --env-file .env python scripts/drift.py
    uv run --env-file .env python scripts/drift.py --recent-days 30 --baseline-days 90
    uv run --env-file .env python scripts/drift.py --out data/private/reports/drift.md

The exit code is 1 when something got worse, so a scheduled run can raise the alarm.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mahjong_api import drift, observability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recent-days", type=int, default=14, help="Length of the recent window")
    parser.add_argument(
        "--baseline-days", type=int, default=60, help="Length of the baseline before it"
    )
    parser.add_argument(
        "--environments",
        default=",".join(drift.ENVIRONMENTS),
        help="Langfuse environments to read, comma-separated",
    )
    parser.add_argument("--out", type=Path, help="Also save the report to this file")
    args = parser.parse_args()

    langfuse = observability.langfuse_client()
    try:
        signed_in = langfuse.auth_check()
    except Exception as e:
        signed_in = False
        print(f"Langfuse: {e}")
    if not signed_in:
        raise SystemExit(
            "Langfuse keys are missing or wrong: set LANGFUSE_PUBLIC_KEY_MJ, "
            "LANGFUSE_SECRET_KEY_MJ and LANGFUSE_BASE_URL (see .env.example)."
        )
    until = datetime.now(UTC)
    split = until - timedelta(days=args.recent_days)
    since = split - timedelta(days=args.baseline_days)
    environments = tuple(args.environments.split(","))
    series, changes = drift.collect(langfuse, since, until, environments)
    findings = drift.compare(series, split, since)
    text = drift.report(findings, changes, since, split, until, environments)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")
    return 1 if any(finding.verdict == "worse" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
