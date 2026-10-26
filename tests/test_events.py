from __future__ import annotations

import json
from pathlib import Path

from agent.events import read_log, reconstruct_conversation
from agent.providers import call, respond
from agent.types import TokenUsage
from tests.fakes import Canned, make_rig, payload


def test_log_records_every_call_result_and_token_count(repo: Path, tmp_path: Path) -> None:
    log = tmp_path / "logs" / "run.jsonl"
    script = [
        respond(call("read_file", {"path": "src/calc.py"}), usage=TokenUsage(1000, 50)),
        respond(call("run_command", {"command": "pytest -q"})),
        respond(call("finish", {"status": "blocked", "summary": "Tests fail."})),
    ]
    rig = make_rig(repo, script, results={"pytest -q": Canned(exit_code=1, stdout="1 failed")}, log_path=log)
    outcome = rig.agent.run("Check the tests.")
    rig.events.close()

    lines = log.read_text(encoding="utf-8").splitlines()
    raw = [json.loads(line) for line in lines]
    assert all({"seq", "ts", "elapsed_s", "agent", "kind", "data"} <= set(r) for r in raw)
    assert [r["seq"] for r in raw] == list(range(1, len(raw) + 1))
    assert raw[0]["ts"].startswith("2026-01-01T00:00:00")

    records = read_log(log)
    kinds = [r.kind for r in records]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    assert kinds.count("model_called") == 3 and kinds.count("tool_executed") == 2

    calls = [payload(r) for r in records if r.kind == "model_called"]
    assert calls[0]["usage"] == {"prompt_tokens": 1000, "completion_tokens": 50}
    assert calls[0]["usage_estimated"] is False and calls[1]["usage_estimated"] is True

    tools = [payload(r) for r in records if r.kind == "tool_executed"]
    assert tools[0]["name"] == "read_file"
    assert tools[0]["effects"]["reads"] == [{"path": "src/calc.py", "first": 1, "last": 2, "total": 2, "truncated": False}]
    command = tools[1]["effects"]["command"]
    assert command["exit_code"] == 1 and command["kind"] == "mutating" and command["stdout_tail"] == "1 failed"

    finished = payload(records[-1])["outcome"]
    assert finished["status"] == "blocked" and finished["message"] == outcome.message

    # The log alone reproduces the conversation exactly.
    assert reconstruct_conversation(records) == rig.agent.messages
    assert any("1 failed" in m.content for m in reconstruct_conversation(records) if m.role == "tool")


def test_unwritable_log_does_not_stop_the_run(repo: Path, tmp_path: Path) -> None:
    blocked_dir = tmp_path / "not-a-dir"
    blocked_dir.write_text("a file where a directory should be", encoding="utf-8")
    rig = make_rig(repo, [respond(call("finish", {"status": "blocked", "summary": "x"}))],
                   log_path=blocked_dir / "run.jsonl")
    assert rig.agent.run("Do it.").cause == "reported"
    assert "cannot open event log" in rig.events.write_error
    assert rig.kinds()[-1] == "run_finished"  # still recorded in memory


def test_undecodable_file_names_do_not_break_the_log(repo: Path, tmp_path: Path) -> None:
    import os

    os.close(os.open(os.fsencode(str(repo)) + b"/caf\xe9.txt", os.O_CREAT | os.O_WRONLY, 0o644))
    log = tmp_path / "run.jsonl"
    script = [respond(call("list_files", {"pattern": "*.txt"})), respond(call("read_file", {"path": "caf\udce9.txt"})),
              respond(call("finish", {"status": "blocked", "summary": "x"}))]
    rig = make_rig(repo, script, log_path=log)
    assert rig.agent.run("List files.").cause == "reported"
    rig.events.close()
    assert rig.events.write_error == ""
    records = read_log(log)
    assert records[-1].kind == "run_finished"
    assert reconstruct_conversation(records) == rig.agent.messages
