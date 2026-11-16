# Showcase copy of services/api/src/mahjong_api/experiments.py from the private mahjong-companion repo. Here to be read, not run.

"""Fixed test sets run as Langfuse experiments, to tell the model drifting from the games
changing (ADR 7).

In real games the inputs change all the time, so a dip in quality or a jump in tokens could be
the model, the prompt or just harder positions. Here the inputs are frozen (``evalset``), stored
as Langfuse datasets, and each run is an experiment on them; Langfuse lines the runs up side by
side, so a change between two runs is the system, not the input.

- ``strategist-positions``: the strategist answers nine made-up positions. Each run records the
  picks, their tokens, cost and latency, the checks that need no judgment, and (when a local
  judge is running) the judge's verdicts.
- ``judge-calibration``: the local judge grades hand-labelled answers. Each run records how often
  it agreed with the labels, so a judge model that got worse after an update shows up.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langfuse import Evaluation

from mahjong_api import evalset, judge, strategist

STRATEGIST_DATASET = "strategist-positions"
JUDGE_DATASET = "judge-calibration"


def strategist_items() -> list[dict[str, Any]]:
    """One dataset item per position: what the strategist is told, and the engine's closest
    hand as the reference answer."""
    card = evalset.sample_card()
    items = []
    for position in evalset.POSITIONS.values():
        closest = evalset.candidates(card, position.state())[0]
        items.append(
            {
                "id": f"{STRATEGIST_DATASET}-{position.name}",
                "input": {
                    "position": position.name,
                    "players": position.players,
                    "dealer": position.dealer,
                    "tiles": list(position.tiles),
                    "events": [dict(event) for event in position.events],
                },
                "expected_output": {
                    "closest_hand": closest.hand.id,
                    "tiles_needed": closest.needed,
                },
                "metadata": {"about": position.about},
            }
        )
    return items


def judge_items() -> list[dict[str, Any]]:
    """One dataset item per hand-labelled answer: what the judge is shown and the verdicts a
    careful person gives."""
    return [
        {
            "id": f"{JUDGE_DATASET}-{labelled.name}",
            "input": evalset.judge_material(labelled),
            "expected_output": labelled.expect,
            "metadata": {"case": labelled.name, "position": labelled.position, "why": labelled.why},
        }
        for labelled in evalset.JUDGE_CASES
    ]


DATASETS = {
    STRATEGIST_DATASET: (
        "Made-up positions on the sample card that the strategist answers in every regression "
        "run. Frozen inputs: a change between runs is the model or the prompt.",
        strategist_items,
    ),
    JUDGE_DATASET: (
        "Strategist answers, good and deliberately broken, with hand-written verdicts. "
        "Measures how well a local judge model agrees with a careful person.",
        judge_items,
    ),
}


def sync_dataset(langfuse: Any, name: str) -> int:
    """Create or update a dataset from the code. Items keep their IDs, so earlier runs stay
    comparable. Returns how many items were written."""
    description, build = DATASETS[name]
    langfuse.create_dataset(name=name, description=description)
    items = build()
    for item in items:
        langfuse.create_dataset_item(dataset_name=name, **item)
    return len(items)


def run_name(prefix: str) -> str:
    return f"{prefix} {datetime.now(UTC):%Y-%m-%d %H:%M}"


def _judged(output: dict[str, Any], situation: str, candidates: list[dict[str, Any]]) -> Any:
    return judge.Case(
        trace_id="experiment",
        situation=situation,
        candidates=candidates,
        hand_id=output["hand_id"],
        label=output["label"],
        reason=output["reason"],
        tools=list(output.get("tools") or []),
        engine_rank=output.get("engine_rank"),
    )


def run_strategist(
    langfuse: Any, claude: Any, backend: Any | None = None, name: str | None = None
) -> Any:
    """Ask the strategist about every position and record the run as an experiment.

    ``claude`` is an ``anthropic.Anthropic``: this costs about a cent a position. With
    ``backend`` (a running local judge) every answer is judged as well.
    """
    card = evalset.sample_card()
    name = name or run_name("strategist")
    session = f"eval-{name.replace(' ', '-').replace(':', '')}"

    def table(position: str) -> tuple[Any, list[Any]]:
        """The game state of a named position and the hands the strategist chooses among."""
        state = evalset.POSITIONS[position].state()
        return state, evalset.candidates(card, state)

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        state, hands = table(item.input["position"])
        pick = strategist.ask(claude, card, state, session, evalset.reads(card, state))
        return {
            "hand_id": pick.hand_id,
            "label": pick.label,
            "reason": pick.reason,
            "engine_rank": [odds.hand.id for odds in hands].index(pick.hand_id) + 1,
            "tools": list(pick.tools),
        }

    def checks(*, output: dict[str, Any], expected_output: dict[str, Any], **_: Any) -> list:
        words = len(output["reason"].split())
        return [
            Evaluation(
                name="picked_closest",
                value=output["hand_id"] == expected_output["closest_hand"],
                data_type="BOOLEAN",
                comment=f"picked {output['hand_id']}; closest {expected_output['closest_hand']}",
            ),
            Evaluation(name="pick_engine_rank", value=output["engine_rank"]),
            Evaluation(
                name="reason_within_limit",
                value=words <= strategist.REASON_WORDS,
                data_type="BOOLEAN",
                comment=f"{words} words",
            ),
        ]

    def judged(*, input: dict[str, Any], output: dict[str, Any], **_: Any) -> list:
        state, hands = table(input["position"])
        case = _judged(
            output, strategist._situation(state, hands), [strategist._odds(odds) for odds in hands]
        )
        verdict = judge.judge(backend, case, langfuse)
        stamp = {"judge_model": verdict.model, "rubric_version": judge.RUBRIC_VERSION}
        scores = [
            Evaluation(
                name=check.score,
                value=verdict.checks[check.name][0],
                data_type="BOOLEAN",
                comment=verdict.checks[check.name][1][:480] or None,
                metadata=stamp,
            )
            for check in judge.CHECKS
        ]
        return [*scores, Evaluation(name=judge.QUALITY, value=verdict.quality, metadata=stamp)]

    def averages(*, item_results: list[Any], **_: Any) -> list:
        return _averages(
            item_results, ("picked_closest", "reason_within_limit", judge.QUALITY), "avg_"
        )

    return langfuse.get_dataset(STRATEGIST_DATASET).run_experiment(
        name="strategist",
        run_name=name,
        description="The strategist on the fixed positions.",
        task=task,
        evaluators=[checks, judged] if backend is not None else [checks],
        run_evaluators=[averages],
        max_concurrency=2,
        metadata={
            "model": strategist.MODEL,
            "effort": strategist.EFFORT,
            "prompt_version": strategist.PROMPT_VERSION,
            "judge_model": getattr(backend, "model", "none"),
        },
    )


def _averages(item_results: list[Any], names: tuple[str, ...], prefix: str) -> list[Evaluation]:
    """The mean of each named item score over a run, for comparing runs at a glance."""
    out = []
    for name in names:
        values = [
            float(evaluation.value)
            for result in item_results
            for evaluation in result.evaluations
            if evaluation.name == name
        ]
        if values:
            mean = sum(values) / len(values)
            out.append(
                Evaluation(name=f"{prefix}{name}", value=mean, comment=f"{len(values)} items")
            )
    return out


def run_calibration(
    langfuse: Any, backend: Any, repeats: int = 1, name: str | None = None
) -> tuple[Any, judge.Calibration]:
    """Grade the hand-labelled answers with the local judge and record the run as an experiment.
    Returns Langfuse's result and the same agreement figures ``judge.calibrate`` gives."""
    results: list[tuple[str, list[judge.Verdict] | judge.JudgeError, dict[str, bool]]] = []

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        case = judge.Case(trace_id=item.id, **item.input)
        try:
            runs = judge.judge_repeatedly(backend, case, repeats, langfuse)
        except judge.JudgeError as e:
            results.append((item.metadata["case"], e, item.expected_output))
            raise
        results.append((item.metadata["case"], runs, item.expected_output))
        return runs[0].summary()

    def agreement(*, output: dict[str, Any], expected_output: dict[str, bool], **_: Any) -> list:
        agreed = {name: output[name]["pass"] == want for name, want in expected_output.items()}
        wrong = [name for name, same in agreed.items() if not same]
        false_passes = [name for name in wrong if output[name]["pass"]]
        return [
            Evaluation(
                name="judge_agreement",
                value=sum(agreed.values()) / len(agreed),
                comment="agreed on every labelled check"
                if not wrong
                else f"disagreed on: {', '.join(wrong)}",
            ),
            Evaluation(
                name="judge_false_pass",
                value=bool(false_passes),
                data_type="BOOLEAN",
                comment=f"passed a failing answer on: {', '.join(false_passes)}"
                if false_passes
                else None,
            ),
        ]

    def overall(*, item_results: list[Any], **_: Any) -> list:
        return _averages(item_results, ("judge_agreement", "judge_false_pass"), "avg_")

    outcome = langfuse.get_dataset(JUDGE_DATASET).run_experiment(
        name="judge-calibration",
        run_name=name or run_name(f"judge {backend.model.rsplit('/', 1)[-1]}"),
        description="The local judge on hand-labelled answers.",
        task=task,
        evaluators=[agreement],
        run_evaluators=[overall],
        max_concurrency=1,  # one local model, one answer at a time
        metadata={
            "judge_model": backend.model,
            "rubric_version": judge.RUBRIC_VERSION,
            "repeats": str(repeats),
        },
    )
    return outcome, judge.summarise(backend.model, results)
