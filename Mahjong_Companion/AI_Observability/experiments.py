# Showcase copy of services/api/src/mahjong_api/experiments.py from the private mahjong-companion repo. Here to be read, not run.

"""Fixed test sets run as Langfuse experiments, to tell the model drifting from the games
changing (ADR 7).

In real games the inputs change all the time, so a dip in quality or a jump in tokens could be
the model, the prompt or just harder positions. Here the inputs are frozen (``evalset``), stored
as Langfuse datasets, and each run is an experiment on them; Langfuse lines the runs up side by
side, so a change between two runs is the system, not the input.

- ``strategist-positions``: the strategist answers nine made-up positions. Each run records the
  picks, their tokens, cost and latency, and the checks that need no judgment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langfuse import Evaluation

from mahjong_api import evalset, strategist

STRATEGIST_DATASET = "strategist-positions"


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


DATASETS = {
    STRATEGIST_DATASET: (
        "Made-up positions on the sample card that the strategist answers in every regression "
        "run. Frozen inputs: a change between runs is the model or the prompt.",
        strategist_items,
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


def run_strategist(langfuse: Any, claude: Any, name: str | None = None) -> Any:
    """Ask the strategist about every position and record the run as an experiment.

    ``claude`` is an ``anthropic.Anthropic``: this costs about a cent a position.
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

    def averages(*, item_results: list[Any], **_: Any) -> list:
        return _averages(item_results, ("picked_closest", "reason_within_limit"), "avg_")

    return langfuse.get_dataset(STRATEGIST_DATASET).run_experiment(
        name="strategist",
        run_name=name,
        description="The strategist on the fixed positions.",
        task=task,
        evaluators=[checks],
        run_evaluators=[averages],
        max_concurrency=2,
        metadata={
            "model": strategist.MODEL,
            "effort": strategist.EFFORT,
            "prompt_version": strategist.PROMPT_VERSION,
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
