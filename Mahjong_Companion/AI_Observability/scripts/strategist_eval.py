# Showcase copy of scripts/strategist_eval.py from the private mahjong-companion repo. Here to be read, not run.

"""Ask the strategist about the same made-up positions every time, and record the run.

In real games the positions change, so a dip in quality or a jump in tokens could be the model,
the prompt or just a harder game. These nine positions on the made-up sample card never change:
when two runs differ, the system changed. Each run is a Langfuse experiment on the dataset
``strategist-positions``; Langfuse shows the runs side by side with their picks, tokens, cost,
latency and scores.

    uv run --env-file .env python scripts/strategist_eval.py --yes
    uv run --env-file .env python scripts/strategist_eval.py --yes --judge

This calls the Claude API once or more per position: nine positions cost roughly $0.10 a run.
It refuses to run without --yes. With --judge (and Ollama running, see scripts/judge.py) the
local judge grades every answer as well, at no extra cost.

Run it after changing the strategist's prompt or model, and now and then in between: a model
behind a fixed name can change too.
"""

from __future__ import annotations

import argparse

import anthropic
from mahjong_api import evalset, experiments, judge, observability, strategist


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="Agree to the API cost and run")
    parser.add_argument(
        "--judge", action="store_true", help="Grade each answer with the local judge"
    )
    parser.add_argument("--name", help="Name for this run (default: strategist + date and time)")
    args = parser.parse_args()
    count = len(evalset.POSITIONS)
    if not args.yes:
        print(
            f"{count} positions. This calls {strategist.MODEL} at least once for each "
            "(roughly $0.10 in all); add --yes."
        )
        return 1

    observability.use_environment("eval")
    langfuse = observability.langfuse_client()
    if not langfuse.auth_check():
        raise SystemExit("Langfuse keys are missing or wrong (see .env.example).")
    backend = None
    if args.judge:
        backend = judge.Ollama()
        if problem := backend.problem():
            raise SystemExit(problem)

    experiments.sync_dataset(langfuse, experiments.STRATEGIST_DATASET)
    result = experiments.run_strategist(langfuse, anthropic.Anthropic(), backend, args.name)
    observability.flush()

    print(f"Run: {result.run_name}")
    for item in result.item_results:
        output = item.output if isinstance(item.output, dict) else {}
        scores = {evaluation.name: evaluation.value for evaluation in item.evaluations}
        position = item.item.input["position"]
        if not output:
            print(f"  {position}: no answer")
            continue
        closest = "closest hand" if scores.get("picked_closest") else "not the closest hand"
        graded = f", judge {scores[judge.QUALITY]:.0%}" if judge.QUALITY in scores else ""
        print(f"  {position}: {output['hand_id']} ({closest}{graded}): {output['reason']}")
    for evaluation in result.run_evaluations:
        print(f"{evaluation.name}: {evaluation.value:.2f}")
    if result.experiment_url:
        print(f"Compare with earlier runs: {result.experiment_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
