# Showcase copy of scripts/judge.py from the private mahjong-companion repo. Here to be read, not run.

"""Grade the strategist's picks with a local open model, and check the model is fit to grade.

The judge is an open model that Ollama serves on this machine, so it costs nothing per call and
your game data goes nowhere new. Get it once (about 8 GB; for an 8 GB graphics card set
JUDGE_MODEL=hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf instead):

    ollama pull hf.co/google/gemma-4-12B-it-qat-q4_0-gguf

Then, with the Langfuse keys in .env:

    uv run --env-file .env python scripts/judge.py check        # is everything in place?
    uv run --env-file .env python scripts/judge.py calibrate    # is this model fit to judge?
    uv run --env-file .env python scripts/judge.py run          # grade the new picks

``calibrate`` grades 18 hand-labelled answers (made-up positions, some deliberately wrong) and
shows where the judge disagrees with the labels. Run it before trusting a new judge model and
after updating one; add --record to keep each run in Langfuse (dataset ``judge-calibration``),
so a judge that got worse shows up. ``run`` reads the strategist's picks of the last 30 days
from Langfuse, skips those this model and rubric already graded, and writes five scores onto
each pick's trace: judge_grounded, judge_matches_pick, judge_sound_choice, judge_clear and
judge_quality.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from mahjong_api import experiments, judge, observability

BAR = 0.8  # a check must agree with the labels this often for the model to count as fit


def backend_from(args: argparse.Namespace) -> judge.Ollama:
    return judge.Ollama(model=args.model, url=args.url, num_ctx=args.num_ctx)


def langfuse_or_exit():
    langfuse = observability.langfuse_client()
    try:
        ok = langfuse.auth_check()
    except Exception as e:
        ok = False
        print(f"Langfuse: {e}")
    if not ok:
        raise SystemExit(
            "Langfuse keys are missing or wrong: set LANGFUSE_PUBLIC_KEY_MJ, "
            "LANGFUSE_SECRET_KEY_MJ and LANGFUSE_BASE_URL (see .env.example)."
        )
    return langfuse


def check(args: argparse.Namespace) -> int:
    backend = backend_from(args)
    problem = backend.problem()
    print(f"Judge model: {backend.model}")
    print(f"Ollama at {backend.url}: {problem or 'ready'}")
    print(f"Rubric version: {judge.RUBRIC_VERSION} ({len(judge.CHECKS)} checks)")
    try:
        langfuse_or_exit()
        print("Langfuse: signed in")
    except SystemExit as e:
        print(e)
        return 1
    return 1 if problem else 0


def show_calibration(result: judge.Calibration, cases: int, repeats: int, bar: float) -> int:
    print(f"\nJudge {result.model}, rubric {judge.RUBRIC_VERSION}: {cases} answers x {repeats}")
    print(f"{'check':<14}{'agreed':>8}{'kappa':>8}{'passed a fail':>15}{'failed a pass':>15}")
    for name, found in result.checks.items():
        kappa = "-" if found.kappa is None else f"{found.kappa:.2f}"
        agreed = f"{found.agreed}/{found.labelled}"
        print(f"{name:<14}{agreed:>8}{kappa:>8}{found.false_pass:>15}{found.false_fail:>15}")
    steady = (
        ""
        if result.consistency is None
        else f", same verdict on every repeat: {result.consistency:.0%}"
    )
    print(f"Overall agreement {result.agreement:.0%}{steady}, {result.seconds:.0f} s of judging.")
    for line in result.failed_cases:
        print(f"  NO VERDICT {line}")
    if result.disagreements:
        print("Where the judge and the labels differ:")
        for line in result.disagreements:
            print(f"  {line}")
    if result.fit(bar):
        print(f"\nFit to judge: every check agrees with the labels at least {bar:.0%} of the time.")
        return 0
    print(
        f"\nNot fit to judge at the {bar:.0%} bar. Try a larger model (JUDGE_MODEL), or read the "
        "disagreements: a label or the rubric may be what needs fixing."
    )
    return 2


def calibrate(args: argparse.Namespace) -> int:
    backend = backend_from(args)
    if problem := backend.problem():
        raise SystemExit(problem)
    cases = judge.labelled_cases()
    if args.record:
        langfuse = langfuse_or_exit()
        experiments.sync_dataset(langfuse, experiments.JUDGE_DATASET)
        outcome, result = experiments.run_calibration(langfuse, backend, args.repeats)
        observability.flush()
        print(f"Recorded in Langfuse: {outcome.experiment_url or outcome.run_name}")
    else:

        def progress(name: str, runs, expected) -> None:
            if isinstance(runs, judge.JudgeError):
                print(f"  {name}: no verdict ({runs})")
                return
            wrong = [c for c, want in expected.items() if runs[0].checks[c][0] != want]
            print(f"  {name}: {'agrees' if not wrong else 'differs on ' + ', '.join(wrong)}")

        result = judge.calibrate(backend, cases, args.repeats, on_case=progress)
    return show_calibration(result, len(cases), args.repeats, args.bar)


def run(args: argparse.Namespace) -> int:
    backend = backend_from(args)
    langfuse = langfuse_or_exit()
    now = datetime.now(UTC)
    since = now - timedelta(days=args.days)
    environments = tuple(args.environments.split(","))
    cases = judge.cases_from_langfuse(langfuse, since, environments)
    done = set() if args.rejudge else judge.already_judged(langfuse, since, backend.model)
    todo = [case for case in cases if case.trace_id not in done]
    if args.limit:
        todo = todo[-args.limit :]  # the most recent ones
    print(
        f"{len(cases)} strategist picks since {since:%Y-%m-%d} in {', '.join(environments)}; "
        f"{len(cases) - len([c for c in cases if c.trace_id not in done])} already graded by "
        f"this model and rubric; grading {len(todo)}."
    )
    if args.dry_run or not todo:
        for case in todo:
            print(f"  would grade {case.started:%Y-%m-%d %H:%M} {case.hand_id}: {case.reason}")
        return 0
    if problem := backend.problem():
        raise SystemExit(problem)
    run_id = f"judge-{now:%Y%m%d-%H%M%S}"
    graded, total = 0, 0.0
    for case in todo:
        try:
            verdict = judge.judge_and_score(langfuse, backend, case, run_id)
        except judge.JudgeError as e:
            print(f"  {case.started:%Y-%m-%d %H:%M} {case.hand_id}: no verdict ({e})")
            continue
        graded += 1
        total += verdict.quality
        failed = f"failed {', '.join(verdict.failed)}" if verdict.failed else "passed all four"
        print(f"  {case.started:%Y-%m-%d %H:%M} {case.hand_id}: {failed} ({verdict.seconds:.0f} s)")
    observability.flush()
    if graded:
        print(
            f"Graded {graded} picks, average quality {total / graded:.0%}. Scores are in Langfuse."
        )
    return 0 if graded == len(todo) else 2


def main() -> int:
    judge_options = argparse.ArgumentParser(add_help=False)
    judge_options.add_argument(
        "--model", help=f"Ollama model (default: JUDGE_MODEL or {judge.DEFAULT_MODEL})"
    )
    judge_options.add_argument(
        "--url", help=f"Ollama address (default: JUDGE_URL or {judge.DEFAULT_URL})"
    )
    judge_options.add_argument(
        "--num-ctx", type=int, default=judge.NUM_CTX, help="Judge context, tokens"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check",
        parents=[judge_options],
        help="Is Ollama running, the model pulled, Langfuse reachable?",
    )
    calibration = commands.add_parser(
        "calibrate", parents=[judge_options], help="Grade the hand-labelled answers"
    )
    calibration.add_argument("--repeats", type=int, default=1, help="Grade each answer this often")
    calibration.add_argument("--bar", type=float, default=BAR, help="Agreement needed per check")
    calibration.add_argument("--record", action="store_true", help="Keep the run in Langfuse")
    grading = commands.add_parser(
        "run", parents=[judge_options], help="Grade the strategist's picks in Langfuse"
    )
    grading.add_argument("--days", type=int, default=30, help="How far back to look")
    grading.add_argument("--limit", type=int, help="Grade at most this many, newest first")
    grading.add_argument("--rejudge", action="store_true", help="Grade picks already graded too")
    grading.add_argument("--dry-run", action="store_true", help="List what would be graded")
    grading.add_argument(
        "--environments",
        default=",".join(judge.ENVIRONMENTS),
        help="Langfuse environments to grade, comma-separated",
    )
    args = parser.parse_args()
    observability.use_environment("judge")
    return {"check": check, "calibrate": calibrate, "run": run}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
