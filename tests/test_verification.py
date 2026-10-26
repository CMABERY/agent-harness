"""The verification gate: `done` needs a successful, non-read-only command after the last edit."""

from __future__ import annotations

from pathlib import Path

from agent.providers import call, respond
from agent.types import ToolCall
from tests.fakes import Canned, make_rig

FIX = call("edit_file", {"path": "src/calc.py", "old_string": "a - b", "new_string": "a + b"})
DONE = call("finish", {"status": "done", "summary": "Fixed add()."})


def run_command(command: str) -> ToolCall:
    return call("run_command", {"command": command})


def test_done_without_any_command_after_the_edit_is_downgraded(repo: Path) -> None:
    rig = make_rig(repo, [respond(FIX), respond(DONE)])
    outcome = rig.agent.run("Fix add().")
    assert (outcome.status, outcome.cause) == ("blocked", "unverified")
    assert "There is no passing verification after the last file modification (src/calc.py, step 1)" \
        in outcome.message
    assert "No command was run after it" in outcome.message
    assert "The agent's summary was: Fixed add()." in outcome.message
    assert outcome.files_modified == ("src/calc.py",) and outcome.evidence == ()
    assert "verification_downgraded" in rig.kinds()


def test_verification_before_the_last_edit_does_not_count(repo: Path) -> None:
    script = [respond(FIX), respond(run_command("pytest -q")),
              respond(call("edit_file", {"path": "src/calc.py", "old_string": "a + b", "new_string": "b + a"})),
              respond(DONE)]
    rig = make_rig(repo, script)
    outcome = rig.agent.run("Fix add().")
    assert outcome.cause == "unverified" and "step 3" in outcome.message


def test_failed_command_after_the_edit_is_named(repo: Path) -> None:
    rig = make_rig(repo, [respond(FIX), respond(run_command("pytest -q")), respond(DONE)],
                   results={"pytest -q": Canned(exit_code=1)})
    outcome = rig.agent.run("Fix add().")
    assert outcome.cause == "unverified"
    assert "`pytest -q` failed after it (step 2, exit 1) and was not re-run successfully" in outcome.message


def test_read_only_commands_are_not_verification(repo: Path) -> None:
    rig = make_rig(repo, [respond(FIX), respond(run_command("cat src/calc.py")), respond(run_command("echo ok")),
                          respond(DONE)])
    outcome = rig.agent.run("Fix add().")
    assert outcome.cause == "unverified" and "read-only" in outcome.message


def test_successful_check_after_the_last_edit_passes_with_evidence(repo: Path) -> None:
    script = [respond(FIX), respond(run_command("pytest -q")), respond(run_command("git diff")), respond(DONE)]
    rig = make_rig(repo, script, results={"pytest -q": Canned(stdout="5 passed")})
    outcome = rig.agent.run("Fix add().")
    assert (outcome.status, outcome.cause) == ("done", "reported")
    assert [(e.step, e.command, e.output_tail) for e in outcome.evidence] == [(2, "pytest -q", "5 passed")]


def test_nothing_modified_means_nothing_to_verify(repo: Path) -> None:
    rig = make_rig(repo, [respond(call("read_file", {"path": "src/calc.py"})),
                          respond(call("finish", {"status": "done", "summary": "add() subtracts."}))])
    outcome = rig.agent.run("What does add() do?")
    assert outcome.status == "done" and outcome.files_modified == () and outcome.evidence == ()


def test_a_failing_check_is_not_rescued_by_a_later_unrelated_success(repo: Path) -> None:
    script = [respond(FIX), respond(run_command("pytest -q")), respond(run_command("mkdir -p build")), respond(DONE)]
    rig = make_rig(repo, script, results={"pytest -q": Canned(exit_code=1)})
    outcome = rig.agent.run("Fix add().")
    assert outcome.cause == "unverified"
    assert "`pytest -q` failed after it (step 2, exit 1)" in outcome.message


def test_rerunning_the_failing_check_successfully_resolves_it(repo: Path) -> None:
    script = [respond(FIX), respond(run_command("pytest -q")), respond(run_command("pytest -q")), respond(DONE)]
    rig = make_rig(repo, script, results={"pytest -q": [Canned(exit_code=1), Canned(stdout="1 passed")]})
    outcome = rig.agent.run("Fix add().")
    assert outcome.status == "done"
    assert [(e.step, e.command) for e in outcome.evidence] == [(3, "pytest -q")]


def test_a_check_with_a_leading_comment_still_counts(repo: Path) -> None:
    command = "# run the unit tests\npython -m unittest -q"
    rig = make_rig(repo, [respond(FIX), respond(run_command(command)), respond(DONE)],
                   results={command: Canned(stderr="OK")})
    assert rig.agent.run("Fix add().").status == "done"


def test_a_masked_exit_status_is_not_a_passing_check(repo: Path) -> None:
    masked = "pytest -q || true"
    rig = make_rig(repo, [respond(FIX), respond(run_command(masked)), respond(DONE)],
                   results={masked: Canned(stdout="1 failed")})
    outcome = rig.agent.run("Fix add().")
    assert outcome.cause == "unverified" and "its exit status is masked" in outcome.message


def test_a_rerun_in_another_directory_does_not_resolve_a_failure(repo: Path) -> None:
    (repo / "pkg").mkdir()
    (repo / "docs").mkdir()
    script = [respond(FIX), respond(call("run_command", {"command": "make test", "cwd": "pkg"})),
              respond(call("run_command", {"command": "make test", "cwd": "docs"})), respond(DONE)]
    rig = make_rig(repo, script, handler=lambda command, cwd: Canned(exit_code=1 if cwd.name == "pkg" else 0))
    outcome = rig.agent.run("Fix add().")
    assert outcome.cause == "unverified" and "`make test` failed after it" in outcome.message


def test_a_swallowed_setup_failure_does_not_mask_the_final_check(repo: Path) -> None:
    command = "pip install -q -e . || true; pytest -q"
    rig = make_rig(repo, [respond(FIX), respond(run_command(command)), respond(DONE)],
                   results={command: Canned(stdout="1 passed")})
    assert rig.agent.run("Fix add().").status == "done"
