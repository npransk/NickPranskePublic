# Showcase copy of scripts/langfuse_smoke.py from the private mahjong-companion repo. Here to be read, not run.

"""Check that tracing works end to end, then clean up.

    uv run --env-file .env python scripts/langfuse_smoke.py

Sends one made-up action per feature (strategist, photo reading, table talk, play-by-play)
through the app's real code to your Langfuse project, with scripted stand-ins for Claude and
Jev, so it costs nothing. It then reads the traces back and checks that they arrived in the
right session and environment, tagged, versioned, linked to their prompt, priced once and
scored.

Everything it sends is filed under the ``selftest`` environment, which no dashboard or report
reads. Each run leaves about 10 observations and 18 scores there: Langfuse is asked to delete
the traces but only queues that request.

Run it after changing keys or the tracing code, not on a schedule. It takes about a minute.
"""

from __future__ import annotations

import os

from mahjong_api import observability, selftest


def main() -> int:
    # Always its own environment, whatever this machine normally traces under.
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = selftest.ENVIRONMENT
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
    print("Sending one made-up action per feature, then reading it back (about a minute)...")
    checks, clean_up = selftest.run(langfuse)
    for check in checks:
        note = "" if check.passed or not check.detail else f" ({check.detail})"
        print(f"  {'ok  ' if check.passed else 'FAIL'} {check.name}{note}")
    failed = [check for check in checks if not check.passed]
    print("Tracing works." if not failed else f"{len(failed)} check(s) failed.")
    print(clean_up)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
