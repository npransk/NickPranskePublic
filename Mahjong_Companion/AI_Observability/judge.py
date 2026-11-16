# Showcase copy of services/api/src/mahjong_api/judge.py from the private mahjong-companion repo. Here to be read, not run.

"""A local open model judges the strategist's picks (ADR 7).

The player's "I'll go for it" tap says whether a pick was followed, not whether its reason was
true. This judge reads what the strategist saw (your tiles, the engine's numbers, any tool
results) and its answer, and gives four pass/fail verdicts, each with the evidence it found:

- ``grounded``: every number and tile the reason states is true.
- ``matches_pick``: the reason argues for the hand that was picked.
- ``sound_choice``: the pick is the closest hand, or the reason gives a real case for another.
- ``clear``: one sentence to the player that names a hand and backs it with a tile or number.

Why it is built this way:

- **A different model family from the one being judged.** Claude writes the pick; an open model
  served by Ollama on your own machine judges it, so the judge shares none of Claude's habits and
  costs nothing per call. The app in Azure is too small to host a model, so judging is a batch
  job (``scripts/judge.py``) that reads traces from Langfuse and writes scores back.
- **Pass or fail on one thing at a time**, with the evidence written before the verdict. Small
  models are steadier on four narrow yes/no checks than on one 1-10 grade.
- **The judge is measured too.** ``calibrate`` runs it on hand-labelled answers
  (``evalset.JUDGE_CASES``), including deliberately broken ones, and reports how often it agrees.
  Run it before trusting a new judge model, and again after updating one.
- **Verdicts are versioned.** Each score carries the judge model and a fingerprint of the rubric,
  so a change in scores can be told apart from a change of judge.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mahjong_api import history
from mahjong_api.observability import (
    marking_errors,
    prompt_link,
    prompt_version,
    record_score,
    trace,
)

# Gemma 4 12B, Google's own 4-bit build: about 8 GB, fits a 12 GB card with room for context.
# On an 8 GB card use hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf (set JUDGE_MODEL).
DEFAULT_MODEL = "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf"
DEFAULT_URL = "http://localhost:11434"
NUM_CTX = 8192  # Ollama's default context can be too short for a full situation
SEED = 7
ATTEMPTS = 2
PROMPT_NAME = "judge-strategist"
# Traces written before the app was told apart by environment are "default".
ENVIRONMENTS = ("production", "default")


@dataclass(frozen=True)
class Check:
    name: str
    score: str  # the Langfuse score it becomes
    rule: str


CHECKS = (
    Check(
        "grounded",
        "judge_grounded",
        "Every number and every tile fact the reason states is true of the picked hand, the "
        "player's tiles or the table as shown, and none breaks a rule above. Check each number "
        "against the picked hand's own row (tiles_needed, outs) and each tile the reason says "
        'the player holds against "Your tiles". Opinions ("looks good") are not facts. A reason '
        "that states no facts passes.\n"
        '   pass: the reason says "you need only 5 tiles" and the picked hand\'s tiles_needed '
        "is 5.\n"
        '   fail: it says "3 tiles away" but tiles_needed is 6; it mentions "your four '
        "flowers\" but the player holds one; it cites an opponent's discard that is not in "
        "the data; it gives another hand's numbers as the picked hand's.",
    ),
    Check(
        "matches_pick",
        "judge_matches_pick",
        "The hand the reason tells the player to go for is the picked hand. Compare the hand "
        "name or tiles in the reason with the picked hand's label and target. A reason that "
        "names no hand passes.\n"
        "   fail: the picked hand's target is four sets of 9s, but the reason says to collect a "
        "run of 1s, 2s and 3s.",
    ),
    Check(
        "sound_choice",
        "judge_sound_choice",
        "Pass if the picked hand has the lowest tiles_needed of all the candidates (a tie "
        'counts; "Facts worked out for you" says whether it does). If another candidate needs '
        "fewer tiles, pass only if the reason names a "
        "specific advantage of the picked hand, or a specific problem with the closer hand, "
        "that the data confirms: more outs, the closer hand is concealed, tiles the closer "
        "hand needs are already discarded. Otherwise fail. Do not weigh whether the advantage "
        "is big enough.",
    ),
    Check(
        "clear",
        "judge_clear",
        "The reason is one sentence addressed to the player, says which hand to go for, and "
        "backs it with at least one specific tile or number. Fail if it is vague "
        '("it\'s the best"), hedges between options, or runs to several sentences.',
    ),
)
QUALITY = "judge_quality"

SYSTEM = (
    """You check the work of an assistant that advises one player ("you", Player 1) in a game \
of American Mahjong.

How the assistant works: a rules engine lists up to five candidate hands, closest first. For \
each it gives tiles_needed (tiles still missing: fewer is better), outs (live tiles that would \
help: more is better), concealed (true means the hand may not call discards, which makes it \
harder), joker_swaps, and target (the tiles of the finished hand: "3x 5 bam" is three 5 bams). \
The assistant picks one candidate and gives the player a one-sentence reason. When it looked \
something up with a tool, the result is shown.

Rules you may need: a joker can stand in only for a tile in a set of three or more identical \
tiles, never for a single tile or a pair. There are four of every tile. "Soap" is the white \
dragon.

Judge the answer on four checks. Each is pass or fail. Use only the data you are shown; do not \
bring in your own ideas about strategy. For each check, first write the evidence in one or two \
sentences, quoting the reason and the data it is checked against, then give the verdict.

"""
    + "\n".join(f"{n}. {check.name}: {check.rule}" for n, check in enumerate(CHECKS, 1))
    + "\n\nAnswer with the JSON object only."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        check.name: {
            "type": "object",
            # Evidence comes first so the verdict follows from it, not the other way round.
            "properties": {"evidence": {"type": "string"}, "pass": {"type": "boolean"}},
            "required": ["evidence", "pass"],
            "additionalProperties": False,
        }
        for check in CHECKS
    },
    "required": [check.name for check in CHECKS],
    "additionalProperties": False,
}

# Changes whenever the instructions, the checks or the answer shape do; on every verdict.
RUBRIC_VERSION = prompt_version(SYSTEM, SCHEMA)


class JudgeError(RuntimeError):
    """The judge couldn't give a usable verdict."""


@dataclass(frozen=True)
class Case:
    """One strategist answer and everything it was based on."""

    trace_id: str
    situation: str
    candidates: list[dict[str, Any]]
    hand_id: str
    label: str
    reason: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    engine_rank: int | None = None
    session_id: str | None = None
    environment: str | None = None
    started: datetime | None = None


@dataclass(frozen=True)
class Verdict:
    checks: dict[str, tuple[bool, str]]  # check -> (passed, evidence)
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def quality(self) -> float:
        """Share of the checks passed."""
        return sum(passed for passed, _ in self.checks.values()) / len(self.checks)

    @property
    def failed(self) -> list[str]:
        return [name for name, (passed, _) in self.checks.items() if not passed]

    def summary(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            **{name: {"pass": p, "evidence": e} for name, (p, e) in self.checks.items()},
        }


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


class Ollama:
    """The local model server. ``JUDGE_MODEL`` and ``JUDGE_URL`` (or Ollama's own
    ``OLLAMA_HOST``) override the defaults. ``JUDGE_THINK=false`` (or ``true``) is passed on as
    Ollama's ``think`` setting, for a reasoning model whose thinking gets in the way of its
    JSON answer; unset, the model's own default applies."""

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        num_ctx: int = NUM_CTX,
        timeout: float = 600,
    ) -> None:
        self.model = model or os.environ.get("JUDGE_MODEL") or DEFAULT_MODEL
        url = url or os.environ.get("JUDGE_URL") or os.environ.get("OLLAMA_HOST") or DEFAULT_URL
        self.url = (url if "://" in url else f"http://{url}").rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        think = os.environ.get("JUDGE_THINK", "").strip().lower()
        self.think = {"true": True, "false": False}.get(think)
        # The model server is on this machine: never send it through a web proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _call(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def problem(self) -> str | None:
        """None when the server is up and has the model; otherwise what to do about it."""
        try:
            models = self._call("/api/tags").get("models", [])
        except (OSError, ValueError) as e:
            return (
                f"Ollama isn't answering at {self.url} ({e}). Install it from ollama.com and "
                "start it (`ollama serve`), or set JUDGE_URL."
            )
        names = {m.get("name") for m in models} | {m.get("model") for m in models}
        if not {self.model, f"{self.model}:latest"} & names:
            return f"The judge model isn't on this machine yet. Run: ollama pull {self.model}"
        return None

    def chat(self, system: str, user: str, schema: dict[str, Any], seed: int = SEED) -> Reply:
        """One answer in the given JSON shape, as repeatable as the model allows."""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "seed": seed, "num_ctx": self.num_ctx},
        }
        if self.think is not None:
            body["think"] = self.think
        try:
            answer = self._call("/api/chat", body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise JudgeError(f"Ollama refused the request ({e.code}): {detail}") from e
        except (OSError, ValueError) as e:
            raise JudgeError(f"Ollama didn't answer: {e}") from e
        input_tokens = answer.get("prompt_eval_count") or 0
        if input_tokens >= self.num_ctx - 8:
            # Ollama cuts a prompt that doesn't fit, from the top: the rubric would be gone.
            raise JudgeError(
                f"The case filled the judge's {self.num_ctx}-token context; raise --num-ctx"
            )
        return Reply(
            text=(answer.get("message") or {}).get("content") or "",
            model=answer.get("model") or self.model,
            input_tokens=input_tokens,
            output_tokens=answer.get("eval_count") or 0,
            seconds=(answer.get("total_duration") or 0) / 1e9,
        )


def _held(situation: str) -> str | None:
    """The player's tiles counted ("3x 3 bam, 2x flower"), from the situation's first line."""
    found = re.search(r"Your tiles \(\d+\): ([^\n]*)", situation)
    if not found:
        return None
    counts: dict[str, int] = {}
    for tile in found.group(1).split(","):
        if tile.strip():
            counts[tile.strip()] = counts.get(tile.strip(), 0) + 1
    return ", ".join(f"{n}x {tile}" for tile, n in counts.items())


def question(case: Case) -> str:
    """Everything the judge is shown about one answer. The counting is done for it (tiles
    held, which candidate is closest): small models judge better than they count."""
    lines = ["## What the assistant was given", case.situation, ""]
    lines.append("## What the assistant looked up")
    if case.tools:
        for tool in case.tools:
            asked = json.dumps(tool.get("input") or {})
            lines.append(f"- {tool['name']}({asked}) returned: {json.dumps(tool.get('output'))}")
    else:
        lines.append("Nothing.")

    lines += ["", "## Facts worked out for you"]
    if held := _held(case.situation):
        lines.append(f"The player's tiles, counted: {held}.")
    needs = [(row.get("tiles_needed"), row.get("hand_id")) for row in case.candidates]
    needs = [(needed, hand_id) for needed, hand_id in needs if isinstance(needed, int)]
    row = next((c for c in case.candidates if c.get("hand_id") == case.hand_id), None)
    if needs and row is not None and isinstance(row.get("tiles_needed"), int):
        lowest = min(needed for needed, _ in needs)
        closest = [hand_id for needed, hand_id in needs if needed == lowest]
        if row["tiles_needed"] == lowest:
            lines.append(
                f"The picked hand needs {lowest} tiles, the fewest of the {len(needs)} candidates."
            )
        else:
            lines.append(
                f"The picked hand needs {row['tiles_needed']} tiles. A closer candidate exists: "
                f"{', '.join(closest)} needs {lowest}."
            )

    lines += ["", "## The assistant's answer", f'Picked hand: {case.hand_id} ("{case.label}")']
    if row is not None:
        lines.append(f"The picked hand's row from the engine: {json.dumps(row)}")
    lines += [f'Reason given to the player: "{case.reason}"', "", "Judge this answer."]
    return "\n".join(lines)


def parse(text: str) -> dict[str, tuple[bool, str]]:
    """The four verdicts from the model's answer. Tolerates text around the JSON and verdicts
    spelled as words; anything else is a ``JudgeError``."""
    try:
        data = json.loads(text)
    except ValueError:
        found = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(found.group(0)) if found else None
        except ValueError:
            data = None
    if not isinstance(data, dict):
        raise JudgeError(f"The judge's answer wasn't JSON: {text[:200]!r}")
    checks = {}
    for check in CHECKS:
        entry = data.get(check.name)
        verdict = entry.get("pass") if isinstance(entry, dict) else None
        if isinstance(verdict, str):
            verdict = {"true": True, "pass": True, "false": False, "fail": False}.get(
                verdict.strip().lower()
            )
        if not isinstance(verdict, bool):
            raise JudgeError(f"The judge gave no pass/fail for {check.name}: {text[:200]!r}")
        checks[check.name] = (verdict, str(entry.get("evidence") or "").strip())
    return checks


class _NoObservation:
    def update(self, **fields: Any) -> None:
        pass


@contextmanager
def _generation(langfuse: Any | None, backend: Any, messages: list[dict]) -> Iterator[Any]:
    if langfuse is None:
        yield _NoObservation()
        return
    with (
        langfuse.start_as_current_observation(
            name="judge",
            as_type="generation",
            model=backend.model,
            version=RUBRIC_VERSION,
            model_parameters={"temperature": 0, "num_ctx": getattr(backend, "num_ctx", None)},
            input=messages,
        ) as generation,
        marking_errors(generation),
    ):
        yield generation


def judge(backend: Any, case: Case, langfuse: Any | None = None, seed: int = SEED) -> Verdict:
    """Ask the local model for its four verdicts on one answer. With ``langfuse``, the call is
    recorded as a generation (the judge's own tokens and time are worth watching too)."""
    user = question(case)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    last: JudgeError | None = None
    for attempt in range(ATTEMPTS):
        with _generation(langfuse, backend, messages) as generation:
            reply = backend.chat(SYSTEM, user, SCHEMA, seed + attempt)
            generation.update(
                output=reply.text,
                usage_details={"input": reply.input_tokens, "output": reply.output_tokens},
                metadata={
                    "attempt": attempt + 1,
                    "seed": seed + attempt,
                    "served_model": reply.model,
                    "seconds": round(reply.seconds, 2),
                },
            )
            try:
                checks = parse(reply.text)
            except JudgeError as e:
                # A malformed answer: try once more with another seed.
                generation.update(level="WARNING", status_message=str(e)[:300])
                last = e
                continue
            # The model as configured, not as the server spells it back, so that verdicts
            # from one judge always carry one name.
            return Verdict(
                checks, backend.model, reply.input_tokens, reply.output_tokens, reply.seconds
            )
    raise last or JudgeError("The judge gave no verdict")


def judge_and_score(langfuse: Any, backend: Any, case: Case, run_id: str) -> Verdict:
    """Judge one traced answer and write the verdicts onto its trace as scores.

    The judging is a trace of its own (session = this judging run); the scores land on the
    trace that was judged, in that trace's environment, one of each per trace: judging again
    overwrites.
    """
    with (
        trace(
            PROMPT_NAME,
            game_id=run_id,
            feature="judge",
            version=RUBRIC_VERSION,
            prompt=prompt_link(PROMPT_NAME, RUBRIC_VERSION),
            judged_trace=case.trace_id,
            judge_model=backend.model,
        ),
        langfuse.start_as_current_observation(
            name=PROMPT_NAME,
            as_type="evaluator",
            input={
                "judged_trace": case.trace_id,
                "hand_id": case.hand_id,
                "reason": case.reason,
                "engine_rank": case.engine_rank,
            },
        ) as evaluator,
        marking_errors(evaluator),
    ):
        judge_trace = langfuse.get_current_trace_id()
        verdict = judge(backend, case, langfuse)
        evaluator.update(output=verdict.summary())
    stamp = {
        "judge_model": verdict.model,
        "rubric_version": RUBRIC_VERSION,
        "judge_trace": judge_trace,
    }
    for check in CHECKS:
        passed, evidence = verdict.checks[check.name]
        record_score(
            langfuse,
            check.score,
            passed,
            trace_id=case.trace_id,
            score_id=f"{case.trace_id}-{check.score}",
            comment=evidence[:480] or None,
            metadata=stamp,
            environment=case.environment,
        )
    passed_count = len(CHECKS) - len(verdict.failed)
    record_score(
        langfuse,
        QUALITY,
        verdict.quality,
        trace_id=case.trace_id,
        score_id=f"{case.trace_id}-{QUALITY}",
        comment=f"{passed_count} of {len(CHECKS)} checks passed"
        + (f"; failed: {', '.join(verdict.failed)}" if verdict.failed else ""),
        metadata=stamp,
        environment=case.environment,
    )
    return verdict


def _candidates_in(situation: str) -> list[dict[str, Any]]:
    """The engine's candidate list as written into the situation text."""
    found = re.search(r"closest first:\s*(\[.*\])\s*Which one", situation, re.S)
    try:
        rows = json.loads(found.group(1)) if found else []
    except ValueError:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def cases_from_langfuse(
    langfuse: Any, since: datetime, environments: tuple[str, ...] = ENVIRONMENTS
) -> list[Case]:
    """Every strategist ask since ``since`` that ended in a pick, oldest first."""
    scope = {"since": since, "environment": list(environments), "fields": "core,basic,io"}
    agents = [
        (o, history.loads(o.input), history.loads(o.output))
        for o in history.observations(langfuse, name="strategist", type="AGENT", **scope)
    ]
    agents = [(o, i, out) for o, i, out in agents if isinstance(out, dict) and out.get("reason")]
    wanted = {o.trace_id for o, _, _ in agents}
    if not wanted:
        return []

    tools: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for o in history.observations(langfuse, type="TOOL", **scope):
        if o.trace_id in wanted and o.name in ("hand_details", "opponent_reads"):
            call = {
                "name": o.name,
                "input": history.loads(o.input),
                "output": history.loads(o.output),
            }
            tools[o.trace_id].append((o.start_time, call))

    # Traces from before the situation was put on the agent: it is the first message to Claude.
    legacy = {o.trace_id for o, i, _ in agents if not (isinstance(i, dict) and i.get("situation"))}
    first_message: dict[str, tuple[datetime, str]] = {}
    if legacy:
        for o in history.observations(langfuse, type="GENERATION", name="claude", **scope):
            messages = history.loads(o.input)
            if o.trace_id not in legacy or not isinstance(messages, list):
                continue
            text = next(
                (
                    m.get("content")
                    for m in messages
                    if isinstance(m, dict) and m.get("role") == "user"
                ),
                None,
            )
            earliest = first_message.get(o.trace_id)
            if isinstance(text, str) and (earliest is None or o.start_time < earliest[0]):
                first_message[o.trace_id] = (o.start_time, text)

    cases = []
    for o, given, picked in agents:
        situation = given.get("situation") if isinstance(given, dict) else None
        situation = situation or first_message.get(o.trace_id, (None, ""))[1]
        if not situation:
            continue  # nothing to judge the reason against
        rows = given.get("candidates") if isinstance(given, dict) else None
        if not (rows and isinstance(rows[0], dict)):
            rows = _candidates_in(situation)
        order = [row.get("hand_id") for row in rows]
        row = next((r for r in rows if r.get("hand_id") == picked.get("hand_id")), {})
        cases.append(
            Case(
                trace_id=o.trace_id,
                situation=situation,
                candidates=rows,
                hand_id=picked["hand_id"],
                label=picked.get("label") or row.get("label") or picked["hand_id"],
                reason=picked["reason"],
                tools=[call for _, call in sorted(tools[o.trace_id], key=lambda t: t[0])],
                engine_rank=picked.get("engine_rank")
                or (order.index(picked["hand_id"]) + 1 if picked["hand_id"] in order else None),
                session_id=o.session_id,
                environment=o.environment,
                started=o.start_time,
            )
        )
    return sorted(cases, key=lambda case: case.started)


def already_judged(langfuse: Any, since: datetime, model: str) -> set[str]:
    """Traces that already carry a verdict from this judge model and this rubric."""
    done = set()
    for score in history.scores(langfuse, since=since, names=[QUALITY]):
        stamp = getattr(score, "metadata", None) or {}
        kind, trace_id = history.subject(score)
        same = stamp.get("rubric_version") == RUBRIC_VERSION and stamp.get("judge_model") == model
        if kind == "trace" and trace_id and same:
            done.add(trace_id)
    return done


@dataclass(frozen=True)
class CheckAgreement:
    """How one check's verdicts compared with the hand labels."""

    labelled: int = 0
    agreed: int = 0
    false_pass: int = 0  # the judge passed an answer labelled as failing: the costly mistake
    false_fail: int = 0
    kappa: float | None = None  # agreement beyond chance; None when it can't be worked out

    @property
    def agreement(self) -> float:
        return self.agreed / self.labelled if self.labelled else 0.0


@dataclass(frozen=True)
class Calibration:
    model: str
    checks: dict[str, CheckAgreement]
    consistency: float | None  # share of verdicts that came out the same on every repeat
    disagreements: list[str]
    failed_cases: list[str]  # the judge gave no usable verdict
    seconds: float = 0.0

    @property
    def agreement(self) -> float:
        labelled = sum(c.labelled for c in self.checks.values())
        return sum(c.agreed for c in self.checks.values()) / labelled if labelled else 0.0

    def fit(self, bar: float) -> bool:
        """Good enough to trust: every check agrees with the labels at least ``bar`` of the
        time, and every case got a verdict."""
        return not self.failed_cases and all(
            c.agreement >= bar for c in self.checks.values() if c.labelled
        )


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for (label, verdict) pairs: 1 is perfect, 0 is what chance would give."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(label == verdict for label, verdict in pairs) / n
    label_yes = sum(label for label, _ in pairs) / n
    verdict_yes = sum(verdict for _, verdict in pairs) / n
    chance = label_yes * verdict_yes + (1 - label_yes) * (1 - verdict_yes)
    return None if chance == 1 else (observed - chance) / (1 - chance)


def judge_repeatedly(
    backend: Any, case: Case, repeats: int = 1, langfuse: Any | None = None
) -> list[Verdict]:
    """Judge one answer ``repeats`` times with different seeds, to see how steady it is."""
    return [judge(backend, case, langfuse, seed=SEED + 10 * repeat) for repeat in range(repeats)]


def summarise(
    model: str, results: list[tuple[str, list[Verdict] | JudgeError, dict[str, bool]]]
) -> Calibration:
    """Compare verdicts with hand labels. ``results`` holds, per labelled answer, its name, the
    verdicts of each repeat (or the error that stopped it) and the expected verdicts. A check
    missing from an answer's labels is a judgment call and isn't counted."""
    pairs: dict[str, list[tuple[bool, bool]]] = {check.name: [] for check in CHECKS}
    disagreements, failed, steady, seen, seconds, repeated = [], [], 0, 0, 0.0, False
    for name, runs, expected in results:
        if isinstance(runs, JudgeError):
            failed.append(f"{name}: {runs}")
            continue
        model = runs[0].model
        repeated = repeated or len(runs) > 1
        seconds += sum(run.seconds for run in runs)
        for check in CHECKS:
            seen += 1
            steady += len({run.checks[check.name][0] for run in runs}) == 1
            if check.name not in expected:
                continue
            label, (verdict, evidence) = expected[check.name], runs[0].checks[check.name]
            pairs[check.name].append((label, verdict))
            if label != verdict:
                said = "passed" if verdict else "failed"
                disagreements.append(f"{name} / {check.name}: the judge {said} it. {evidence}")
    checks = {
        name: CheckAgreement(
            labelled=len(found),
            agreed=sum(label == verdict for label, verdict in found),
            false_pass=sum(verdict and not label for label, verdict in found),
            false_fail=sum(label and not verdict for label, verdict in found),
            kappa=kappa(found),
        )
        for name, found in pairs.items()
    }
    consistency = steady / seen if repeated and seen else None
    return Calibration(model, checks, consistency, disagreements, failed, seconds)


def calibrate(
    backend: Any,
    cases: list[tuple[str, Case, dict[str, bool]]],
    repeats: int = 1,
    on_case: Any = None,
) -> Calibration:
    """Run the judge on labelled answers (name, case, expected verdicts) and compare."""
    results: list[tuple[str, list[Verdict] | JudgeError, dict[str, bool]]] = []
    for name, case, expected in cases:
        try:
            runs: list[Verdict] | JudgeError = judge_repeatedly(backend, case, repeats)
        except JudgeError as e:
            runs = e
        results.append((name, runs, expected))
        if on_case:
            on_case(name, runs, expected)
    return summarise(backend.model, results)


def labelled_cases() -> list[tuple[str, Case, dict[str, bool]]]:
    """The hand-labelled answers in ``evalset``, as cases the judge can read."""
    from mahjong_api import evalset

    out = []
    for labelled in evalset.JUDGE_CASES:
        material = evalset.judge_material(labelled)
        case = Case(trace_id=f"calibration-{labelled.name}", **material)
        out.append((labelled.name, case, labelled.expect))
    return out
