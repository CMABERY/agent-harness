"""End to end: a fixture repository with a failing test, fixed by a scripted model.

Everything is real except the model and the clock: files are read and edited
on disk, and the tests run in a real subprocess of the current interpreter
through SubprocessRunner. The scripted model's arguments are fixed in
advance, so the assertions are about what the harness did with them.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from agent.context import CharTokenCounter
from agent.events import EventLog, read_log, reconstruct_conversation
from agent.host import SubprocessRunner
from agent.loop import Agent, Runtime
from agent.policy import DefaultPolicy
from agent.providers import ScriptedProvider, call, respond
from agent.workspace import Workspace
from tests.fakes import FakeClock, payload

MODULE = '''"""Small statistics helpers."""


def mean(values):
    """Arithmetic mean of a non-empty sequence."""
    return sum(values) / (len(values) - 1)
'''

TESTS = '''import unittest

from stats import mean


class MeanTest(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_single_value(self):
        self.assertEqual(mean([5]), 5)


if __name__ == "__main__":
    unittest.main()
'''


def test_agent_fixes_a_failing_test_end_to_end(tmp_path: Path) -> None:
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / "stats.py").write_text(MODULE, encoding="utf-8")
    (repo / "test_stats.py").write_text(TESTS, encoding="utf-8")
    test_command = f"{shlex.quote(sys.executable)} -m unittest -q"

    script = [
        respond(call("list_files", {"pattern": "**/*.py"}), text="Let me look at the repository."),
        respond(call("read_file", {"path": "stats.py"}), call("read_file", {"path": "test_stats.py"})),
        respond(call("run_command", {"command": test_command})),
        respond(call("update_plan", {
            "acceptance_criteria": [f"`{test_command}` passes"],
            "steps": [{"title": "Fix the divisor in mean()", "status": "in_progress"},
                      {"title": "Run the test suite", "status": "pending"}],
            "notes": "mean() divides by len - 1; both tests fail (one with ZeroDivisionError).",
        })),
        respond(call("edit_file", {"path": "stats.py", "old_string": "sum(values) / (len(values) - 1)",
                                   "new_string": "sum(values) / len(values)"})),
        respond(call("run_command", {"command": test_command})),
        respond(call("finish", {"status": "done",
                                "summary": "mean() divided by len(values) - 1; it now divides by len(values).",
                                "evidence": [f"{test_command}: 2 tests OK"]})),
    ]
    clock = FakeClock()
    log_path = tmp_path / "run.jsonl"
    events = EventLog(log_path, clock)
    workspace = Workspace(repo)
    runtime = Runtime(clock, SubprocessRunner(), DefaultPolicy(workspace.root), events, CharTokenCounter())
    agent = Agent(ScriptedProvider(script), workspace, runtime)

    outcome = agent.run("The tests in test_stats.py fail. Fix the bug in stats.py so they pass.")
    events.close()

    # Outcome
    assert (outcome.status, outcome.cause) == ("done", "reported"), outcome.message
    assert outcome.files_modified == ("stats.py",)
    assert [(e.command, e.exit_code) for e in outcome.evidence] == [(test_command, 0)]
    assert "OK" in outcome.evidence[0].output_tail
    assert outcome.iterations == 7

    # Files on disk
    assert (repo / "stats.py").read_text(encoding="utf-8") == MODULE.replace("(len(values) - 1)", "len(values)")
    assert (repo / "test_stats.py").read_text(encoding="utf-8") == TESTS

    # Event log
    records = read_log(log_path)
    kinds = [r.kind for r in records]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    assert kinds.count("model_called") == 7
    executed = [payload(r) for r in records if r.kind == "tool_executed"]
    assert [e["name"] for e in executed] == [
        "list_files", "read_file", "read_file", "run_command", "update_plan", "edit_file", "run_command"]
    exit_codes = [e["effects"]["command"]["exit_code"] for e in executed if e["name"] == "run_command"]
    assert exit_codes == [1, 0]
    first_run = next(e for e in executed if e["name"] == "run_command")["effects"]["command"]
    assert "ZeroDivisionError" in first_run["stderr_tail"]
    assert payload(records[-1])["outcome"]["status"] == "done"
    assert all(payload(r)["usage"]["prompt_tokens"] > 0 for r in records if r.kind == "model_called")
    assert reconstruct_conversation(records) == agent.messages
