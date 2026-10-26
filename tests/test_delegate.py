from __future__ import annotations

from pathlib import Path

from agent.loop import load_prompt
from agent.providers import call, respond
from agent.types import ModelResponse, TokenUsage
from tests.fakes import Canned, make_rig

FIX = call("edit_file", {"path": "src/calc.py", "old_string": "a - b", "new_string": "a + b"})
TEST = call("run_command", {"command": "pytest -q"})


def finish(status: str, summary: str = "ok") -> ModelResponse:
    return respond(call("finish", {"status": status, "summary": summary}))


def test_child_gets_a_fresh_scoped_context_and_returns_a_structured_result(repo: Path) -> None:
    script = [
        respond(call("read_file", {"path": "src/calc.py"})),
        respond(call("delegate", {"task": "Fix add() in src/calc.py.", "context": "The test is tests/test_calc.py.",
                                  "max_iterations": 5})),
        respond(FIX, usage=TokenUsage(100, 10)),  # child
        respond(TEST),  # child
        respond(call("finish", {"status": "done", "summary": "Changed - to +; pytest passes."})),  # child
        respond(call("finish", {"status": "done", "summary": "Delegated the fix; verified by the subagent."})),
    ]
    rig = make_rig(repo, script, system_prompt=None, results={"pytest -q": Canned(stdout="1 passed")})
    outcome = rig.agent.run("Fix add().")

    child_first = rig.provider.requests[2]
    assert [m.role for m in child_first.messages] == ["system", "user"]  # none of the parent's conversation
    assert child_first.messages[0].content.endswith(load_prompt("subagent.md"))
    assert "Fix add() in src/calc.py." in child_first.messages[1].content
    assert "# Context from the delegating agent\nThe test is tests/test_calc.py." in child_first.messages[1].content
    assert "Budget: at most 5 model calls" in child_first.messages[1].content
    assert "delegate" not in {t.name for t in child_first.tools}

    result = rig.tool_messages(5)[-1]
    assert result.startswith("[delegate: ok] Subagent main.sub1 finished with status done (reported) after 3")
    assert "Files it modified: src/calc.py." in result
    assert "`pytest -q` exited 0 at step 3" in result
    # The child's verification satisfies the parent's gate: the workspace is verified.
    assert outcome.status == "done" and outcome.files_modified == ("src/calc.py",)
    assert outcome.usage.prompt_tokens >= 100  # child usage is included in the parent's total
    assert {r.agent for r in rig.events.records} == {"main", "main.sub1"}


def test_child_cannot_delegate(repo: Path) -> None:
    script = [
        respond(call("delegate", {"task": "Investigate."})),
        respond(call("delegate", {"task": "Investigate further."})),  # child tries to delegate
        finish("blocked", "cannot delegate"),  # child
        finish("blocked"),
    ]
    rig = make_rig(repo, script)
    rig.agent.run("Investigate.")
    child_view = rig.provider.requests[2].messages
    assert child_view[-1].content.startswith("[delegate: error unknown_tool]")
    assert rig.tool_messages(3)[-1].startswith("[delegate: error subagent_blocked]")


def test_unverified_child_edit_blocks_both_child_and_parent(repo: Path) -> None:
    script = [
        respond(call("delegate", {"task": "Fix add()."})),
        respond(FIX),  # child edits
        finish("done", "fixed"),  # child claims done without verifying
        finish("done", "the subagent fixed it"),  # parent trusts it blindly
    ]
    rig = make_rig(repo, script)
    outcome = rig.agent.run("Fix add().")
    child_result = rig.tool_messages(3)[-1]
    assert child_result.startswith("[delegate: error subagent_blocked]") and "(unverified)" in child_result
    assert "observed no successful verification after its last edit" in child_result
    assert (outcome.status, outcome.cause) == ("blocked", "unverified")


def test_child_answers_only_for_its_own_edits(repo: Path) -> None:
    script = [
        respond(FIX),  # parent edits, not yet verified
        respond(call("delegate", {"task": "Report what tests exist."})),
        respond(call("list_files", {"pattern": "tests/*"})),  # child investigates only
        finish("done", "One test file."),  # child: nothing modified by it, so done stands
        respond(TEST),
        finish("done", "Fixed and verified."),
    ]
    rig = make_rig(repo, script)
    outcome = rig.agent.run("Fix add().")
    assert rig.tool_messages(4)[-1].startswith("[delegate: ok]")
    assert outcome.status == "done"


def test_child_budget_is_its_own(repo: Path) -> None:
    script = [
        respond(call("delegate", {"task": "Look around.", "max_iterations": 1})),
        respond(call("read_file", {"path": "src/calc.py"})),  # the child's only call
        finish("blocked", "child ran out"),
    ]
    rig = make_rig(repo, script)
    outcome = rig.agent.run("Investigate.")
    assert "status budget_exhausted (iterations)" in rig.tool_messages(2)[-1]
    assert outcome.cause == "reported"
