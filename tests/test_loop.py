"""The loop's control flow: outcomes, budgets, recovery, loop detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.providers import ProviderError, ScriptStep, call, respond
from agent.types import Limits, ModelResponse, TokenUsage, ToolCall
from tests.fakes import Canned, FakeClock, make_rig

FIX = call("edit_file", {"path": "src/calc.py", "old_string": "a - b", "new_string": "a + b"})
TEST = call("run_command", {"command": "pytest -q"})
DONE = call("finish", {"status": "done", "summary": "Fixed add().", "evidence": ["pytest -q: 1 passed"]})


def read(path: str = "src/calc.py") -> ToolCall:
    return call("read_file", {"path": path})


def test_happy_path_returns_done_with_harness_evidence(repo: Path) -> None:
    rig = make_rig(repo, [respond(read()), respond(FIX), respond(TEST), respond(DONE)],
                   results={"pytest -q": Canned(stdout="1 passed")})
    outcome = rig.agent.run("Fix add().")
    assert outcome.status == "done" and outcome.cause == "reported"
    assert outcome.message == "Fixed add()."
    assert outcome.files_modified == ("src/calc.py",)
    assert [(e.command, e.exit_code, e.step) for e in outcome.evidence] == [("pytest -q", 0, 3)]
    assert outcome.claimed_evidence == ("pytest -q: 1 passed",)
    assert outcome.iterations == 4 and outcome.usage.total > 0
    roles = [m.role for m in rig.agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"]
    assert rig.agent.messages[3].tool_call_id == rig.agent.messages[2].tool_calls[0].id


def test_agent_can_report_blocked(repo: Path) -> None:
    rig = make_rig(repo, [respond(call("finish", {"status": "blocked", "summary": "Needs the API spec."}))])
    outcome = rig.agent.run("Implement the API.")
    assert (outcome.status, outcome.cause, outcome.message) == ("blocked", "reported", "Needs the API spec.")


# ------------------------------------------------------------- budgets


def test_iteration_cap(repo: Path) -> None:
    script: list[ScriptStep] = [respond(read(p)) for p in ("src/calc.py", "tests/test_calc.py", "nope.py")]
    rig = make_rig(repo, script, limits=Limits(max_iterations=2))
    outcome = rig.agent.run("Look around.")
    assert (outcome.status, outcome.cause) == ("budget_exhausted", "iterations")
    assert outcome.iterations == 2 and rig.provider.remaining == 1


def test_token_cap_counts_reported_usage(repo: Path) -> None:
    script = [respond(read(), usage=TokenUsage(900, 200)), respond(read("tests/test_calc.py"))]
    rig = make_rig(repo, script, limits=Limits(max_total_tokens=1000))
    outcome = rig.agent.run("Look around.")
    assert (outcome.status, outcome.cause) == ("budget_exhausted", "tokens")
    assert outcome.usage == TokenUsage(900, 200) and "1100 of 1000" in outcome.message


def test_wall_clock_cap_uses_the_injected_clock(repo: Path) -> None:
    rig = make_rig(repo, [respond(TEST), respond(read())], limits=Limits(max_wall_seconds=60),
                   results={"pytest -q": Canned(duration_s=61)})
    outcome = rig.agent.run("Run the tests.")
    assert (outcome.status, outcome.cause) == ("budget_exhausted", "wall_clock")
    assert rig.runner.calls[0][2] == 60  # the command could not have outlived the budget either


# ------------------------------------------------------------ recovery


def test_malformed_arguments_get_a_corrective_message_and_a_retry(repo: Path) -> None:
    broken = respond(call("read_file", '{"path": "src/calc.py"'), read("tests/test_calc.py"))
    rig = make_rig(repo, [broken, respond(read()), respond(call("finish", {"status": "blocked", "summary": "x"}))])
    outcome = rig.agent.run("Read.")
    assert outcome.cause == "reported"
    retry = rig.sent(1)
    assert all(m.role != "assistant" for m in retry)  # the broken reply never entered the conversation
    assert retry[-1].role == "user"
    assert "none of its tool calls ran" in retry[-1].content
    assert "arguments of the tool call 1 of 2 (read_file) are not valid JSON" in retry[-1].content
    assert len(rig.runner.calls) == 0 and "malformed_output" in rig.kinds()
    assert len([m for m in rig.agent.messages if m.role == "tool"]) == 1  # only the retried call ran


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (ModelResponse(text="All done, the bug is fixed!"), "no tool call"),
        (ModelResponse(), "neither text nor a tool call"),
        (ModelResponse(text="Here is the full fi", finish_reason="length"), "cut off at the output-token limit"),
        (respond(call("read_file", "[1, 2]")), "must be a JSON object"),
        (respond(call("", {"path": "x"})), "has no tool name"),
    ],
)
def test_unusable_replies_are_explained(repo: Path, reply: ModelResponse, expected: str) -> None:
    rig = make_rig(repo, [reply, respond(call("finish", {"status": "blocked", "summary": "x"}))])
    rig.agent.run("Do it.")
    assert expected in rig.sent(1)[-1].content


def test_consecutive_malformed_replies_end_the_run(repo: Path) -> None:
    rig = make_rig(repo, [ModelResponse(text="thinking...")] * 3, limits=Limits(max_consecutive_malformed=3))
    outcome = rig.agent.run("Do it.")
    assert (outcome.status, outcome.cause) == ("blocked", "malformed_output")
    assert "3 unusable replies in a row" in outcome.message
    assert "1 more unusable reply in a row will end the run" in rig.sent(2)[-1].content


def test_tool_errors_are_returned_to_the_model(repo: Path) -> None:
    script = [
        respond(call("delete_everything", {}), call("read_file", {"path": "missing.py"}),
                call("edit_file", {"path": "src/calc.py"})),
        respond(call("finish", {"status": "blocked", "summary": "x"})),
    ]
    rig = make_rig(repo, script)
    assert rig.agent.run("Do it.").cause == "reported"
    results = rig.tool_messages(1)
    assert results[0].startswith("[delete_everything: error unknown_tool]") and "Available tools:" in results[0]
    assert results[1].startswith("[read_file: error not_found]")
    assert results[2].startswith("[edit_file: error invalid_arguments]") and "old_string: missing" in results[2]


def test_provider_failure_is_a_blocked_outcome(repo: Path) -> None:
    rig = make_rig(repo, [respond(read()), ProviderError("HTTP 401: bad key")])
    outcome = rig.agent.run("Do it.")
    assert (outcome.status, outcome.cause) == ("blocked", "provider_error")
    assert "HTTP 401" in outcome.message and rig.kinds()[-1] == "run_finished"


def test_harness_bug_still_returns_an_outcome(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = make_rig(repo, [respond(read())])

    def broken(*_: object) -> None:
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(rig.agent.shared.ledger, "record", broken)
    outcome = rig.agent.run("Do it.")
    assert (outcome.status, outcome.cause) == ("blocked", "internal_error")
    assert "ledger exploded" in outcome.message and "internal_error" in rig.kinds()


def test_interrupt_returns_an_outcome(repo: Path) -> None:
    def interrupt(_: object) -> ModelResponse:
        raise KeyboardInterrupt

    rig = make_rig(repo, [interrupt])
    assert rig.agent.run("Do it.").cause == "interrupted"


# ------------------------------------------------------- loop detection


def test_identical_calls_trigger_intervention_then_termination(repo: Path) -> None:
    rig = make_rig(repo, [respond(read()) for _ in range(4)], limits=Limits(loop_repeat_threshold=3))
    outcome = rig.agent.run("Read.")
    assert (outcome.status, outcome.cause) == ("blocked", "loop_detected")
    warning = rig.tool_messages(3)[-1]
    assert warning.startswith("[read_file: error loop_detected]") and "3 times in a row" in warning
    assert [r.data["terminated"] for r in rig.events.records if r.kind == "loop_detected"] == [False, True]


def test_different_arguments_reset_the_repeat_count(repo: Path) -> None:
    script = [respond(read()), respond(read()), respond(read("tests/test_calc.py")), respond(read()),
              respond(read()), respond(call("finish", {"status": "blocked", "summary": "x"}))]
    rig = make_rig(repo, script, limits=Limits(loop_repeat_threshold=3))
    assert rig.agent.run("Read.").cause == "reported"
    assert "loop_detected" not in rig.kinds()


def test_finish_must_be_alone(repo: Path) -> None:
    script = [respond(FIX, DONE), respond(TEST), respond(DONE)]
    rig = make_rig(repo, script)
    outcome = rig.agent.run("Fix it.")
    assert outcome.status == "done"
    first = rig.tool_messages(1)
    assert first[0].startswith("[edit_file: ok]")
    assert first[1].startswith("[finish: error finish_not_alone]")


def test_a_corrective_message_never_follows_another_user_message(repo: Path) -> None:
    rig = make_rig(repo, [ModelResponse(text="Let me think."), respond(call("finish", {"status": "blocked",
                                                                                         "summary": "x"}))])
    rig.agent.run("Do it.")
    retry = rig.sent(1)
    assert [m.role for m in retry] == ["system", "user"]
    assert retry[1].content.startswith(rig.sent(0)[1].content + "\n\n[harness] Your last reply was not accepted.")
    from agent.events import reconstruct_conversation

    assert reconstruct_conversation(rig.events.records) == rig.agent.messages


def test_the_iteration_cap_is_run_wide(repo: Path) -> None:
    script = [
        respond(call("delegate", {"task": "Look around.", "max_iterations": 10})),
        respond(read()), respond(read("tests/test_calc.py")), respond(read("nope.py")),  # child
        respond(read()),
    ]
    rig = make_rig(repo, script, limits=Limits(max_iterations=3))
    outcome = rig.agent.run("Investigate.")
    assert len(rig.provider.requests) == 3  # 1 parent call + 2 child calls; then the run-wide cap stops both
    assert (outcome.status, outcome.cause) == ("budget_exhausted", "iterations")
    assert "Run-wide iteration cap reached (3 model calls" in outcome.message
    assert outcome.iterations == 3


def test_a_model_call_that_outlives_the_wall_clock_is_a_budget_outcome(repo: Path) -> None:
    clock = FakeClock()

    def slow_failure(_: object) -> ModelResponse:
        clock.advance(4000)
        raise ProviderError("read timed out")

    rig = make_rig(repo, [slow_failure], limits=Limits(max_wall_seconds=600), clock=clock)
    outcome = rig.agent.run("Do it.")
    assert (outcome.status, outcome.cause) == ("budget_exhausted", "wall_clock")
    assert "during a model call (read timed out)" in outcome.message


def test_duplicate_call_ids_are_repaired_not_rejected(repo: Path) -> None:
    reply = respond(call("read_file", {"path": "src/calc.py"}, call_id="dup"),
                    call("read_file", {"path": "tests/test_calc.py"}, call_id="dup"))
    rig = make_rig(repo, [reply, respond(call("finish", {"status": "blocked", "summary": "x"}))])
    rig.agent.run("Read.")
    assistant, first, second = rig.sent(1)[2:5]
    assert [c.id for c in assistant.tool_calls] == ["dup", "main_call_1"]
    assert (first.tool_call_id, second.tool_call_id) == ("dup", "main_call_1")
    assert "malformed_output" not in rig.kinds()


def test_identical_calls_in_one_reply_are_refused_but_do_not_end_the_run(repo: Path) -> None:
    same = read()
    script = [respond(same, same, same, same), respond(read("tests/test_calc.py")),
              respond(call("finish", {"status": "blocked", "summary": "x"}))]
    rig = make_rig(repo, script, limits=Limits(loop_repeat_threshold=3))
    assert rig.agent.run("Read.").cause == "reported"
    results = rig.tool_messages(1)
    assert [r.split("]")[0] for r in results] == [
        "[read_file: ok", "[read_file: ok", "[read_file: error loop_detected", "[read_file: error loop_detected"]
    assert all(r.data["terminated"] is False for r in rig.events.records if r.kind == "loop_detected")
