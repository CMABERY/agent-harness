"""The real OpenAI adapter, end to end through the CLI, against a strict local server.

The server enforces the conversation rules a real chat-completions endpoint
enforces (an assistant message with tool_calls must be answered by exactly one
tool message per id before anything else; no orphan tool messages; roles are
valid), and answers with HTTP 400 otherwise. Runs on 127.0.0.1 only, so the
test is offline. It covers the paths most likely to produce an invalid
conversation: a discarded malformed reply, a multi-call reply, and compaction.
"""

from __future__ import annotations

import http.server
import io
import json
import shlex
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent.cli import main


def conversation_errors(messages: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not messages or messages[0].get("role") != "system":
        errors.append("first message must be the system prompt")
    pending: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            errors.append(f"{index}: bad role {role!r}")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                errors.append(f"{index}: tool message for unknown or already-answered id {call_id!r}")
            pending.discard(str(call_id))
            if not isinstance(message.get("content"), str):
                errors.append(f"{index}: tool content must be a string")
            continue
        if pending:
            errors.append(f"{index}: {role} message before tool results for {sorted(pending)}")
            pending.clear()
        if role == "user" and index > 0 and messages[index - 1].get("role") == "user":
            errors.append(f"{index}: two user messages in a row (strict chat templates reject this)")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            ids = [c["id"] for c in calls]
            if len(set(ids)) != len(ids) or not all(ids):
                errors.append(f"{index}: missing or duplicate tool_call ids {ids}")
            if calls and message.get("content") not in (None,) and not isinstance(message.get("content"), str):
                errors.append(f"{index}: bad assistant content")
            pending = set(ids)
    if pending:
        errors.append(f"conversation ends with unanswered tool calls {sorted(pending)}")
    return errors


def tool_call(call_id: str, name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def reply(*calls: dict[str, Any], text: str | None = None) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": list(calls)}


@contextmanager
def strict_server(replies: list[dict[str, Any]]) -> Iterator[tuple[str, list[list[str]]]]:
    violations: list[list[str]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            errors = conversation_errors(body["messages"])
            violations.append(errors)
            if errors:
                payload = json.dumps({"error": {"message": "; ".join(errors)}}).encode()
                status = 400
            else:
                message = replies.pop(0)
                payload = json.dumps({
                    "choices": [{"message": message, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": len(json.dumps(body)) // 4, "completion_tokens": 20},
                }).encode()
                status = 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", violations
    finally:
        server.shutdown()
        server.server_close()


def run(base_url: str, workspace: Path, log: Path, *extra: str) -> tuple[int, str]:
    out = io.StringIO()
    environ = {"AGENT_BASE_URL": base_url, "AGENT_API_KEY": "sk-local", "AGENT_MODEL": "strict-test",
               "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}
    code = main(["Fix the failing tests.", "--workspace", str(workspace), "--log", str(log), *extra],
                environ=environ, stdout=out, stderr=out)
    return code, out.getvalue()


def make_repo(root: Path) -> str:
    root.mkdir()
    (root / "stats.py").write_text("def mean(values):\n    return sum(values) / (len(values) - 1)\n")
    (root / "test_stats.py").write_text(
        "import unittest\nfrom stats import mean\n\n\nclass T(unittest.TestCase):\n"
        "    def test_mean(self):\n        self.assertEqual(mean([2, 4, 6]), 4)\n")
    return f"{shlex.quote(sys.executable)} -m unittest -q"


FIX = {"path": "stats.py", "old_string": "(len(values) - 1)", "new_string": "len(values)"}


def test_full_run_over_http_keeps_the_conversation_valid(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    test = make_repo(repo)
    replies = [
        {"role": "assistant", "content": "I will look at the files first."},  # no tool call: rejected, corrected
        reply(tool_call("a", "read_file", {"path": "stats.py"}), tool_call("b", "read_file", {"path": "test_stats.py"}),
              text="Reading both files."),
        reply(tool_call("c", "run_command", {"command": test})),
        reply(tool_call("d", "edit_file", '{"path": "stats.py", "old_string": ')),  # truncated JSON: discarded
        reply(tool_call("e", "edit_file", FIX)),
        reply(tool_call("f", "run_command", {"command": test})),
        reply(tool_call("g", "finish", {"status": "done", "summary": "Divide by len(values).",
                                        "evidence": ["unittest: OK"]})),
    ]
    with strict_server(replies) as (url, violations):
        code, output = run(url, repo, tmp_path / "run.jsonl")
    assert code == 0, output
    assert violations and all(v == [] for v in violations), violations
    assert len(violations) == 7
    assert "sum(values) / len(values)" in (repo / "stats.py").read_text()


def test_compaction_over_http_keeps_the_conversation_valid(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    test = make_repo(repo)
    (repo / "notes.txt").write_text("".join(f"note {i}: " + "lorem ipsum " * 8 + "\n" for i in range(150)))
    replies = [
        reply(tool_call("a", "read_file", {"path": "notes.txt"})),
        reply(tool_call("b", "read_file", {"path": "notes.txt", "offset": 2})),
        reply(tool_call("c", "read_file", {"path": "notes.txt", "offset": 3})),
        reply(tool_call("d", "edit_file", FIX)),
        reply(tool_call("e", "run_command", {"command": test})),
        reply(tool_call("f", "finish", {"status": "done", "summary": "Fixed."})),
    ]
    with strict_server(replies) as (url, violations):
        code, output = run(url, repo, tmp_path / "run.jsonl", "--context-window", "12000")
    assert code == 0, output
    assert all(v == [] for v in violations), violations
    log = (tmp_path / "run.jsonl").read_text()
    assert '"kind": "context_compacted"' in log


def test_the_strict_checker_rejects_invalid_conversations() -> None:
    system = {"role": "system", "content": "s"}
    call = {"role": "assistant", "content": None, "tool_calls": [tool_call("x", "read_file", {})]}
    assert conversation_errors([system, call]) == ["conversation ends with unanswered tool calls ['x']"]
    assert conversation_errors([system, call, {"role": "user", "content": "u"}])[0].startswith("2: user message before")
    orphan = {"role": "tool", "tool_call_id": "y", "content": "r"}
    assert "unknown or already-answered id 'y'" in conversation_errors([system, orphan])[0]
    assert conversation_errors([system, call, {"role": "tool", "tool_call_id": "x", "content": "r"}]) == []
    user = {"role": "user", "content": "u"}
    assert "two user messages in a row" in conversation_errors([system, user, user])[0]
