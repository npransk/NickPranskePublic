# Showcase copy of services/api/tests/test_judge.py from the private mahjong-companion repo. Here to be read, not run.

"""The local judge, with a stand-in Ollama server and scripted judges: no model, no network."""

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from mahjong_api import evalset, judge

ALL_PASS = {
    check.name: {"evidence": "Checked against the data.", "pass": True} for check in judge.CHECKS
}


def answer(**failing):
    """A judge answer that passes everything except the named checks."""
    verdicts = {name: dict(entry) for name, entry in ALL_PASS.items()}
    for name, evidence in failing.items():
        verdicts[name] = {"evidence": evidence, "pass": False}
    return json.dumps(verdicts)


CASE = judge.Case(
    trace_id="a" * 32,
    situation="Players: 2. Your tiles (13): 3 bam, 3 bam, 3 bam\nCandidate hands ...",
    candidates=[
        {"hand_id": "1999-LIKE-1", "label": "Sample Like Numbers, top line", "tiles_needed": 4},
        {"hand_id": "1999-RUN-1", "label": "Sample Runs, top line", "tiles_needed": 9},
    ],
    hand_id="1999-LIKE-1",
    label="Sample Like Numbers, top line",
    reason="Go for like numbers in 3s because you need only 4 tiles.",
    tools=[{"name": "opponent_reads", "input": {}, "output": {"Player 2": {"discards": []}}}],
    engine_rank=1,
    session_id="game-1",
    environment="production",
)


class Scripted:
    """A judge backend that replays answers and records what it was asked."""

    model = "scripted-judge"
    num_ctx = 8192

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def chat(self, system, user, schema, seed=judge.SEED):
        self.asked.append({"system": system, "user": user, "schema": schema, "seed": seed})
        text = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return judge.Reply(text, "scripted-judge:latest", 900, 120, 1.5)


@pytest.fixture
def ollama():
    """A local server that speaks enough of Ollama's API, and remembers the chat requests."""
    state = {"models": ["judge-model:latest"], "requests": [], "reply": answer(), "tokens": 900}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"models": [{"name": m, "model": m} for m in state["models"]]})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            if request["model"] not in state["models"] + ["judge-model"]:
                self._send(404, {"error": f"model '{request['model']}' not found"})
                return
            self._send(
                200,
                {
                    "model": "judge-model:latest",
                    "message": {"role": "assistant", "content": state["reply"]},
                    "prompt_eval_count": state["tokens"],
                    "eval_count": 150,
                    "total_duration": 2_500_000_000,
                },
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


def test_ollama_is_asked_for_a_repeatable_answer_in_the_rubrics_shape(ollama):
    backend = judge.Ollama(model="judge-model", url=ollama["url"])
    reply = backend.chat("system text", "user text", judge.SCHEMA, seed=11)
    (request,) = ollama["requests"]
    assert request["model"] == "judge-model"
    assert request["stream"] is False
    assert request["format"] == judge.SCHEMA
    assert request["options"] == {"temperature": 0, "seed": 11, "num_ctx": judge.NUM_CTX}
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert reply == judge.Reply(answer(), "judge-model:latest", 900, 150, 2.5)


def test_the_answer_shape_asks_for_evidence_before_each_verdict():
    for check in judge.CHECKS:
        entry = judge.SCHEMA["properties"][check.name]
        assert list(entry["properties"]) == ["evidence", "pass"]
    assert judge.SCHEMA["required"] == ["grounded", "matches_pick", "sound_choice", "clear"]
    assert all(check.name in judge.SYSTEM for check in judge.CHECKS)


def test_it_says_what_is_missing_before_any_judging(ollama):
    assert judge.Ollama(model="judge-model", url=ollama["url"]).problem() is None
    missing = judge.Ollama(model="other-model", url=ollama["url"]).problem()
    assert "ollama pull other-model" in missing
    down = judge.Ollama(model="judge-model", url="http://127.0.0.1:9", timeout=2).problem()
    assert "isn't answering" in down


def test_the_address_and_model_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "my-model")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11500")
    backend = judge.Ollama()
    assert (backend.model, backend.url) == ("my-model", "http://127.0.0.1:11500")
    monkeypatch.setenv("JUDGE_URL", "http://gpu-box:11434/")
    assert judge.Ollama().url == "http://gpu-box:11434"


def test_thinking_is_left_to_the_model_unless_set(ollama, monkeypatch):
    judge.Ollama(model="judge-model", url=ollama["url"]).chat("s", "u", judge.SCHEMA)
    assert "think" not in ollama["requests"][0]
    monkeypatch.setenv("JUDGE_THINK", "false")
    judge.Ollama(model="judge-model", url=ollama["url"]).chat("s", "u", judge.SCHEMA)
    assert ollama["requests"][1]["think"] is False


def test_a_case_that_fills_the_context_is_refused_not_judged_on_a_cut_prompt(ollama):
    ollama["tokens"] = judge.NUM_CTX
    backend = judge.Ollama(model="judge-model", url=ollama["url"])
    with pytest.raises(judge.JudgeError, match="context"):
        backend.chat("system", "user", judge.SCHEMA)


def test_a_model_ollama_does_not_have_is_a_clear_error(ollama):
    backend = judge.Ollama(model="not-pulled", url=ollama["url"])
    with pytest.raises(judge.JudgeError, match="not found"):
        backend.chat("system", "user", judge.SCHEMA)


def test_the_judge_is_shown_the_situation_the_lookups_the_pick_and_the_reason():
    text = judge.question(CASE)
    assert "Your tiles (13): 3 bam, 3 bam, 3 bam" in text
    assert 'opponent_reads({}) returned: {"Player 2": {"discards": []}}' in text
    assert 'Picked hand: 1999-LIKE-1 ("Sample Like Numbers, top line")' in text
    assert '"tiles_needed": 4' in text.split("The picked hand's row")[1]
    assert 'Reason given to the player: "Go for like numbers in 3s' in text
    assert "Nothing." in judge.question(judge.Case(**{**CASE.__dict__, "tools": []}))


def test_the_counting_is_done_for_the_judge():
    """Small models judge better than they count, so the counts are worked out in code."""
    text = judge.question(CASE)
    assert "The player's tiles, counted: 3x 3 bam." in text
    assert "The picked hand needs 4 tiles, the fewest of the 2 candidates." in text
    farther = judge.Case(**{**CASE.__dict__, "hand_id": "1999-RUN-1", "label": "Sample Runs"})
    assert (
        "The picked hand needs 9 tiles. A closer candidate exists: 1999-LIKE-1 needs 4."
        in judge.question(farther)
    )
    _, labelled, _ = judge.labelled_cases()[0]
    assert (
        "The player's tiles, counted: 3x 3 bam, 1x 9 bam, 3x 3 crak, 1x 7 crak, 2x 3 dot, "
        "1x 6 dot, 2x flower." in judge.question(labelled)
    )


def test_the_rubrics_own_examples_are_not_the_answers_it_is_tested_on():
    """An example in the instructions that matches a labelled answer would flatter the judge."""
    for case in evalset.JUDGE_CASES:
        assert case.reason not in judge.SYSTEM
    for phrase in ("2 tiles away", "pair of jokers", "2s, 4s, 6s and 8s"):
        assert phrase not in judge.SYSTEM


def test_verdicts_are_read_from_json_even_with_text_around_it():
    plain = judge.parse(answer(grounded="It says 2 tiles; the data says 4."))
    assert plain["grounded"] == (False, "It says 2 tiles; the data says 4.")
    assert plain["clear"] == (True, "Checked against the data.")
    wrapped = judge.parse(f"Here is my assessment:\n```json\n{answer()}\n```\nDone.")
    assert all(passed for passed, _ in wrapped.values())
    worded = {name: {"evidence": "ok", "pass": "Fail"} for name in ALL_PASS}
    assert not any(passed for passed, _ in judge.parse(json.dumps(worded)).values())


@pytest.mark.parametrize(
    "text",
    [
        "I think it is fine.",
        json.dumps({"grounded": {"evidence": "ok", "pass": True}}),  # three checks missing
        json.dumps({**json.loads(answer()), "clear": {"evidence": "ok", "pass": "maybe"}}),
        "[1, 2, 3]",
    ],
)
def test_an_answer_without_four_verdicts_is_an_error(text):
    with pytest.raises(judge.JudgeError):
        judge.parse(text)


def test_a_verdict_is_the_share_of_checks_passed():
    backend = Scripted(answer(grounded="Wrong number.", clear="Vague."))
    verdict = judge.judge(backend, CASE)
    assert verdict.quality == 0.5
    assert verdict.failed == ["grounded", "clear"]
    assert verdict.model == "scripted-judge"  # as configured, whatever the server calls it
    assert (verdict.input_tokens, verdict.output_tokens, verdict.seconds) == (900, 120, 1.5)
    assert backend.asked[0]["system"] == judge.SYSTEM
    assert backend.asked[0]["schema"] == judge.SCHEMA


def test_a_malformed_answer_is_tried_once_more_with_another_seed():
    backend = Scripted("Sure! The answer looks good to me.", answer())
    assert judge.judge(backend, CASE).quality == 1.0
    assert [ask["seed"] for ask in backend.asked] == [judge.SEED, judge.SEED + 1]
    with pytest.raises(judge.JudgeError):
        judge.judge(Scripted("still not JSON"), CASE)


def test_judging_writes_five_scores_onto_the_judged_trace(langfuse):
    backend = Scripted(answer(sound_choice="Needs 10 tiles against 4 and names no advantage."))
    verdict = judge.judge_and_score(langfuse, backend, CASE, "judge-run-1")
    assert verdict.failed == ["sound_choice"]
    scores = {s["name"]: s for s in langfuse.scores}
    assert set(scores) == {
        "judge_grounded",
        "judge_matches_pick",
        "judge_sound_choice",
        "judge_clear",
        "judge_quality",
    }
    assert scores["judge_sound_choice"]["value"] == 0.0
    assert scores["judge_sound_choice"]["data_type"] == "BOOLEAN"
    assert "names no advantage" in scores["judge_sound_choice"]["comment"]
    assert scores["judge_quality"]["value"] == 0.75
    assert scores["judge_quality"]["comment"] == "3 of 4 checks passed; failed: sound_choice"
    for score in scores.values():
        # On the pick's trace and in its environment, one of each: judging again overwrites.
        assert score["trace_id"] == CASE.trace_id
        assert score["environment"] == "production"
        assert score["score_id"] == f"{CASE.trace_id}-{score['name']}"
        assert score["metadata"] == {
            "judge_model": "scripted-judge",
            "rubric_version": judge.RUBRIC_VERSION,
            "judge_trace": langfuse.trace_id,
        }


def test_the_judging_itself_is_traced_with_its_tokens(langfuse):
    judge.judge_and_score(langfuse, Scripted(answer()), CASE, "judge-run-1")
    (evaluator,) = langfuse.named("judge-strategist")
    assert evaluator["as_type"] == "evaluator"
    assert evaluator["input"]["judged_trace"] == CASE.trace_id
    assert evaluator["output"]["quality"] == 1.0
    (generation,) = langfuse.named("judge")
    assert generation["model"] == "scripted-judge"
    assert generation["version"] == judge.RUBRIC_VERSION
    assert generation["usage_details"] == {"input": 900, "output": 120}
    assert generation["metadata"]["served_model"] == "scripted-judge:latest"
    assert generation["input"][0] == {"role": "system", "content": judge.SYSTEM}


def test_a_judge_that_gives_no_verdict_writes_no_scores(langfuse):
    with pytest.raises(judge.JudgeError):
        judge.judge_and_score(langfuse, Scripted("no json here"), CASE, "judge-run-1")
    assert langfuse.scores == []
    assert langfuse.named("judge-strategist")[0]["level"] == "ERROR"
    assert [g["level"] for g in langfuse.named("judge")] == ["WARNING", "WARNING"]


def observation(trace, name, kind, minute, input=None, output=None, environment="production"):
    return SimpleNamespace(
        id=f"{trace}-{name}-{minute}",
        trace_id=trace,
        name=name,
        type=kind,
        input=json.dumps(input) if input is not None else None,
        output=json.dumps(output) if output is not None else None,
        start_time=datetime(2026, 11, 1, 20, minute, tzinfo=UTC),
        session_id="game-1",
        environment=environment,
    )


class Archive:
    """Stands in for Langfuse's read API: filters by name and type, pages two at a time."""

    def __init__(self, observations=(), scores=()):
        self._observations, self._scores = list(observations), list(scores)
        self.api = SimpleNamespace(
            observations=SimpleNamespace(get_many=self._get_observations),
            scores=SimpleNamespace(get_many=self._get_scores),
        )

    def _page(self, rows, cursor):
        start = int(cursor or 0)
        following = start + 2 if start + 2 < len(rows) else None
        return SimpleNamespace(data=rows[start : start + 2], meta=SimpleNamespace(cursor=following))

    def _get_observations(self, cursor=None, name=None, type=None, environment=None, **_):
        rows = [
            o
            for o in self._observations
            if (name is None or o.name == name)
            and (type is None or o.type == type)
            and (environment is None or o.environment in environment)
        ]
        return self._page(rows, cursor)

    def _get_scores(self, cursor=None, name=None, **_):
        names = name.split(",") if name else None
        return self._page([s for s in self._scores if names is None or s.name in names], cursor)


SITUATION = (
    "Players: 2. Your tiles (13): 3 bam\nCandidate hands from the engine, closest first:\n"
    + json.dumps([{"hand_id": "H-1", "label": "First"}, {"hand_id": "H-2", "label": "Second"}])
    + "\nWhich one should I go for?"
)


def test_picks_are_read_back_with_the_situation_and_lookups_they_were_based_on():
    rows = [{"hand_id": "H-1", "label": "First"}, {"hand_id": "H-2", "label": "Second"}]
    archive = Archive(
        [
            observation(
                "t1",
                "strategist",
                "AGENT",
                5,
                {"situation": SITUATION, "candidates": rows},
                {"hand_id": "H-2", "label": "Second", "reason": "Go for it.", "engine_rank": 2},
            ),
            observation("t1", "opponent_reads", "TOOL", 7, {}, {"Player 2": {}}),
            observation("t1", "hand_details", "TOOL", 6, {"hand_id": "H-2"}, {"hand": "H-2"}),
            observation("t1", "submit_pick", "TOOL", 8, {"hand_id": "X"}, {"error": "Unknown"}),
            # An ask that ended without a pick, and one from an eval run: neither is judged.
            observation(
                "t2", "strategist", "AGENT", 9, {"situation": SITUATION, "candidates": rows}
            ),
            observation(
                "t3",
                "strategist",
                "AGENT",
                3,
                {"situation": SITUATION, "candidates": rows},
                {"hand_id": "H-1", "label": "First", "reason": "Closest."},
                environment="eval",
            ),
            observation("t9", "validate", "TOOL", 1, {"round": 1}, {"problems": []}),
        ]
    )
    (case,) = judge.cases_from_langfuse(archive, datetime(2026, 10, 1, tzinfo=UTC))
    assert case.trace_id == "t1"
    assert (case.hand_id, case.label, case.reason, case.engine_rank) == (
        "H-2",
        "Second",
        "Go for it.",
        2,
    )
    assert case.situation == SITUATION and case.candidates == rows
    assert [tool["name"] for tool in case.tools] == ["hand_details", "opponent_reads"]  # in order
    assert (case.session_id, case.environment) == ("game-1", "production")


def test_older_traces_are_read_from_the_first_message_to_claude():
    """Before the situation was put on the agent, it was only in Claude's first message."""
    archive = Archive(
        [
            observation(
                "old",
                "strategist",
                "AGENT",
                5,
                {"candidates": ["H-1", "H-2"]},
                {"hand_id": "H-2", "reason": "Go for the second."},
                environment="default",
            ),
            observation(
                "old",
                "claude",
                "GENERATION",
                7,
                [{"role": "user", "content": SITUATION}, {"role": "assistant", "content": []}],
                environment="default",
            ),
            observation(
                "old",
                "claude",
                "GENERATION",
                6,
                [{"role": "user", "content": SITUATION}],
                environment="default",
            ),
        ]
    )
    (case,) = judge.cases_from_langfuse(archive, datetime(2026, 10, 1, tzinfo=UTC))
    assert case.situation == SITUATION
    assert case.candidates == [
        {"hand_id": "H-1", "label": "First"},
        {"hand_id": "H-2", "label": "Second"},
    ]
    assert (case.label, case.engine_rank, case.environment) == ("Second", 2, "default")


def score(trace, model, rubric, kind="trace"):
    return SimpleNamespace(
        name="judge_quality",
        metadata={"judge_model": model, "rubric_version": rubric},
        subject={"kind": kind, "id": trace},
    )


def test_picks_already_graded_by_this_judge_and_rubric_are_found():
    archive = Archive(
        scores=[
            score("t1", "scripted-judge", judge.RUBRIC_VERSION),
            score("t2", "another-judge", judge.RUBRIC_VERSION),
            score("t3", "scripted-judge", "0ld0ld0l"),
            score("t4", "scripted-judge", judge.RUBRIC_VERSION),
            score("s1", "scripted-judge", judge.RUBRIC_VERSION, kind="session"),
        ]
    )
    since = datetime(2026, 10, 1, tzinfo=UTC)
    assert judge.already_judged(archive, since, "scripted-judge") == {"t1", "t4"}


def test_kappa_is_agreement_beyond_chance():
    assert judge.kappa([(True, True), (False, False)] * 5) == 1.0
    assert judge.kappa([(True, True), (True, False), (False, True), (False, False)]) == 0.0
    assert judge.kappa([(True, True)] * 4) is None  # no failures to tell apart: undefined
    assert judge.kappa([]) is None


class Careful:
    """Judges the labelled answers the way their labels say (an ideal judge)."""

    model = "careful-judge"
    num_ctx = 8192

    def __init__(self, wrong=()):
        self.expected = {
            f'Reason given to the player: "{case.reason}"': case.expect
            for case in evalset.JUDGE_CASES
        }
        self.wrong = set(wrong)  # (reason fragment, check) pairs to get wrong on purpose
        self.calls = 0

    def chat(self, system, user, schema, seed=judge.SEED):
        self.calls += 1
        labels = next(expect for line, expect in self.expected.items() if line in user)
        verdicts = {}
        for check in judge.CHECKS:
            passed = labels.get(check.name, True)
            if any(fragment in user and check.name == name for fragment, name in self.wrong):
                passed = not passed
            verdicts[check.name] = {"evidence": "As labelled.", "pass": passed}
        return judge.Reply(json.dumps(verdicts), self.model, 1500, 200, 2.0)


def test_a_judge_that_matches_every_label_is_fit():
    result = judge.calibrate(Careful(), judge.labelled_cases())
    assert result.agreement == 1.0 and result.fit(0.8)
    assert result.disagreements == [] and result.failed_cases == []
    assert result.checks["grounded"].labelled == 18
    assert result.checks["sound_choice"].labelled == 15  # three judgment calls aren't counted
    assert all(found.kappa == 1.0 for found in result.checks.values())
    assert result.consistency is None  # one pass: nothing to compare
    assert result.seconds == 36.0


def test_a_judge_that_passes_everything_is_caught_by_the_broken_answers():
    result = judge.calibrate(Scripted(answer()), judge.labelled_cases())
    assert not result.fit(0.8)
    grounded = result.checks["grounded"]
    assert (grounded.false_pass, grounded.false_fail) == (8, 0)
    assert grounded.agreement == pytest.approx(10 / 18)
    assert grounded.kappa == 0.0  # no better than chance
    assert any(line.startswith("threes-wrong-numbers / grounded") for line in result.disagreements)


def test_single_disagreements_are_listed_with_the_judges_evidence():
    backend = Careful(wrong=[("your pair of jokers", "grounded")])
    result = judge.calibrate(backend, judge.labelled_cases())
    assert result.disagreements == [
        "threes-invented-jokers / grounded: the judge passed it. As labelled."
    ]
    assert result.checks["grounded"].false_pass == 1
    assert result.fit(0.8) and not result.fit(0.99)


def test_repeats_show_how_steady_the_verdicts_are_and_progress_is_reported():
    seen = []
    backend = Scripted(answer(), answer(clear="Vague this time."))
    cases = judge.labelled_cases()[:2]
    result = judge.calibrate(backend, cases, repeats=2, on_case=lambda *a: seen.append(a[0]))
    assert seen == ["threes-good", "threes-wrong-numbers"]
    # 8 verdicts; "clear" flipped between repeats on the first answer only.
    assert result.consistency == 7 / 8


def test_an_answer_the_judge_cannot_grade_counts_against_it():
    result = judge.calibrate(Scripted("not json"), judge.labelled_cases()[:3])
    assert len(result.failed_cases) == 3 and not result.fit(0.0)
    assert result.agreement == 0.0


def test_the_labelled_answers_still_match_what_the_engine_says():
    """Labels were written against these numbers; if the engine changes them, the labels need
    another look."""
    card = evalset.sample_card()
    for case in evalset.JUDGE_CASES:
        hands = evalset.candidates(card, evalset.POSITIONS[case.position].state())
        facts = {odds.hand.id: (odds.needed, odds.outs) for odds in hands}
        assert case.hand_id in facts, case.name
        for hand_id, expected in case.relies_on.items():
            assert facts[hand_id] == expected, (case.name, hand_id)


def test_the_labelled_answers_cover_every_way_to_fail():
    names = [case.name for case in evalset.JUDGE_CASES]
    assert len(names) == len(set(names))
    checks = {check.name for check in judge.CHECKS}
    for case in evalset.JUDGE_CASES:
        assert set(case.expect) <= checks, case.name
        assert case.why, case.name
    for check in checks:
        labels = [case.expect[check] for case in evalset.JUDGE_CASES if check in case.expect]
        assert labels.count(False) >= 3 and labels.count(True) >= 3, check
    # Some answers pass everything: a judge that fails everything must be caught too.
    assert (
        sum(all(case.expect.values()) and len(case.expect) == 4 for case in evalset.JUDGE_CASES)
        >= 5
    )


def test_a_labelled_answer_becomes_a_full_case_for_the_judge():
    name, case, expected = judge.labelled_cases()[8]
    assert name == "winds-false-opponent-claim" and expected["grounded"] is False
    assert case.engine_rank == 1 and case.label == "Sample Like Numbers, bottom line"
    assert [tool["name"] for tool in case.tools] == ["opponent_reads"]
    assert case.tools[0]["output"]["Player 2"]["discards"] == []
    assert len(case.candidates) == 5 and "Your tiles (13)" in case.situation


def test_the_rubric_version_follows_the_rubric():
    assert judge.RUBRIC_VERSION == judge.prompt_version(judge.SYSTEM, judge.SCHEMA)
    assert judge.RUBRIC_VERSION != judge.prompt_version(judge.SYSTEM + " ", judge.SCHEMA)
