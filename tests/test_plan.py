from __future__ import annotations

from pathlib import Path

from agent.providers import call, respond
from agent.types import JSONObject, ToolCall
from tests.fakes import Canned, make_rig

TEST = call("run_command", {"command": "pytest -q"})
STOP = respond(call("finish", {"status": "blocked", "summary": "stop"}))


def plan(*steps: tuple[str, str], notes: str = "") -> ToolCall:
    payload: JSONObject = {
        "acceptance_criteria": ["`pytest -q` passes"],
        "steps": [{"title": title, "status": status} for title, status in steps],
        "notes": notes,
    }
    return call("update_plan", payload)


def edit(old: str, new: str) -> ToolCall:
    return call("edit_file", {"path": "src/calc.py", "old_string": old, "new_string": new})


def test_plan_is_recorded_and_validated(repo: Path) -> None:
    script = [
        respond(plan(("A", "in_progress"), ("B", "in_progress"))),
        respond(call("update_plan", {"acceptance_criteria": [], "steps": []})),
        respond(plan(("Reproduce", "done"), ("Fix", "in_progress"), ("Verify", "pending"))),
        STOP,
    ]
    rig = make_rig(repo, script)
    rig.agent.run("Fix add().")
    invalid, empty, good = (rig.tool_messages(i)[-1] for i in (1, 2, 3))
    assert invalid.startswith("[update_plan: error invalid_plan]")
    assert empty.startswith("[update_plan: error invalid_arguments]") and "at least 1 item" in empty
    assert good.startswith("[update_plan: ok] Plan v1 recorded: 3 step(s) — 1 done, 1 in progress, 1 pending")
    assert rig.agent.plan.plan is not None and rig.agent.plan.plan.current_step() == "Fix"
    assert "plan_updated" in rig.kinds()


def test_two_failed_attempts_force_a_replan(repo: Path) -> None:
    script = [
        respond(plan(("Fix add()", "in_progress"))),
        respond(TEST),  # reproduction before any edit: not an attempt
        respond(edit("a - b", "b - a")),
        respond(TEST),  # attempt 1 fails
        respond(TEST),  # re-run without editing: not an attempt
        respond(edit("b - a", "abs(a - b)")),
        respond(TEST),  # attempt 2 fails -> replan required
        respond(edit("abs(a - b)", "a + b")),  # refused
        respond(call("read_file", {"path": "src/calc.py"})),  # still allowed
        respond(plan(("Fix add()", "in_progress"))),  # unchanged: rejected
        respond(plan(("Fix add()", "failed"), ("Re-read the test and fix the operator", "in_progress"),
                     notes="Expected 2 + 3 == 5; we keep subtracting.")),
        respond(edit("abs(a - b)", "a + b")),  # allowed again
        STOP,
    ]
    rig = make_rig(repo, script, results={"pytest -q": Canned(exit_code=1, stdout="1 failed")})
    rig.agent.run("Fix add().")

    second_failure = rig.tool_messages(7)[-1]
    assert "[harness note] That was failed attempt 2 at 'Fix add()'" in second_failure
    assert "will be refused until you call update_plan with a revised plan" in second_failure
    assert rig.tool_messages(4)[-1].count("[harness note]") == 0  # attempt 1 only counts, no note
    refused = rig.tool_messages(8)[-1]
    assert refused.startswith("[edit_file: error replan_required]")
    assert rig.tool_messages(9)[-1].startswith("[read_file: ok]")
    assert rig.tool_messages(10)[-1].startswith("[update_plan: error plan_unchanged]")
    assert "replan requirement is cleared" in rig.tool_messages(11)[-1]
    assert rig.tool_messages(12)[-1].startswith("[edit_file: ok]")
    assert (repo / "src" / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    replans = [r.data for r in rig.events.records if r.kind == "replan_required"]
    assert replans == [{"step": "Fix add()", "failed_attempts": 2}]


def test_failures_without_a_plan_require_writing_one(repo: Path) -> None:
    script = [respond(edit("a - b", "b - a")), respond(TEST), respond(edit("b - a", "a * b")), respond(TEST),
              respond(edit("a * b", "a + b")), STOP]
    rig = make_rig(repo, script, results={"pytest -q": Canned(exit_code=1)})
    rig.agent.run("Fix add().")
    assert "until you write a plan with update_plan (you have none yet)" in rig.tool_messages(4)[-1]
    assert rig.tool_messages(5)[-1].startswith("[edit_file: error replan_required]")


def test_a_passing_setup_command_does_not_rescue_a_failing_attempt(repo: Path) -> None:
    install = call("run_command", {"command": "pip install -e ."})
    script = [
        respond(plan(("Fix add()", "in_progress"))),
        respond(edit("a - b", "b - a")), respond(install), respond(TEST),  # attempt 1 fails
        respond(edit("b - a", "a * b")), respond(install), respond(TEST),  # attempt 2 fails -> replan
        STOP,
    ]
    rig = make_rig(repo, script, results={"pytest -q": Canned(exit_code=1), "pip install -e .": Canned()})
    rig.agent.run("Fix add().")
    assert "That was failed attempt 2 at 'Fix add()'" in rig.tool_messages(7)[-1]


def test_changing_only_a_status_is_not_a_new_plan(repo: Path) -> None:
    script = [
        respond(plan(("Fix add()", "in_progress"))),
        respond(edit("a - b", "b - a")), respond(TEST),
        respond(edit("b - a", "a * b")), respond(TEST),
        respond(plan(("Fix add()", "pending"))),  # same step, different status
        STOP,
    ]
    rig = make_rig(repo, script, results={"pytest -q": Canned(exit_code=1)})
    rig.agent.run("Fix add().")
    assert rig.tool_messages(6)[-1].startswith("[update_plan: error plan_unchanged]")
    assert rig.agent.plan.replan_reason


def test_a_subagent_does_not_inherit_the_parents_edits_as_its_attempts(repo: Path) -> None:
    script = [
        respond(edit("a - b", "b - a")),  # parent edits
        respond(call("delegate", {"task": "Investigate the failing test."})),
        respond(TEST),  # child: reproduction before any edit of its own, not an attempt
        respond(call("edit_file", {"path": "tests/test_calc.py", "old_string": "== 5", "new_string": "== 5  # sum"})),
        respond(TEST),  # child: attempt 1 fails
        respond(call("finish", {"status": "blocked", "summary": "still failing"})),
        STOP,
    ]
    rig = make_rig(repo, script, results={"pytest -q": Canned(exit_code=1)})
    rig.agent.run("Fix add().")
    child_view = rig.provider.requests[5].messages
    assert "[harness note]" not in child_view[-1].content  # one failed attempt: no replan demanded yet
    assert "replan_required" not in rig.kinds()
