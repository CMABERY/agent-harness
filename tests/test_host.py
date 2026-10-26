"""The real subprocess runner. These tests start processes (the current Python
interpreter only), but their outcomes do not depend on timing: the timeout
case uses a child that would never finish on its own."""

from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path

import pytest

from agent.host import ProcessResult, SubprocessRunner, child_environment

PY = shlex.quote(sys.executable)


def run(tmp_path: Path, code: str, *, timeout: float = 30, cap: int = 16_000,
        environ: dict[str, str] | None = None) -> ProcessResult:
    runner = SubprocessRunner(environ if environ is not None else {"PATH": "/usr/bin:/bin"})
    return runner.run(f"{PY} -c {shlex.quote(code)}", cwd=tmp_path, timeout_s=timeout, max_output_bytes=cap)


def test_exit_code_and_both_streams(tmp_path: Path) -> None:
    result = run(tmp_path, "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)")
    assert result.exit_code == 3 and not result.timed_out
    assert result.stdout.text == "out\n" and result.stderr.text == "err\n"


def test_output_is_capped_keeping_head_and_tail(tmp_path: Path) -> None:
    result = run(tmp_path, "print('BEGIN' + 'x' * 50000 + 'END')", cap=1000)
    stdout = result.stdout
    assert stdout.truncated and stdout.total_bytes == 50009
    assert stdout.text.startswith("BEGIN") and stdout.text.endswith("END\n")
    assert "bytes omitted" in stdout.text and len(stdout.text) < 1100


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "survived"
    runner = SubprocessRunner({"PATH": "/usr/bin:/bin"})
    command = (f"{PY} -c 'import time; time.sleep(30); open(\"{marker}\", \"w\")' & "
               f"{PY} -c 'import time; print(\"started\", flush=True); time.sleep(30)'")
    result = runner.run(command, cwd=tmp_path, timeout_s=1.0, max_output_bytes=1000)
    assert result.timed_out and result.exit_code is None
    assert result.stdout.text == "started\n"
    assert not marker.exists()


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state not in {"Z", "X"}  # a killed orphan may linger as a zombie until init reaps it


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_background_process_is_killed_when_the_command_returns(tmp_path: Path) -> None:
    runner = SubprocessRunner({"PATH": "/usr/bin:/bin"})
    spawn = f"{PY} -c 'import os, time; open(\"pid\", \"w\").write(str(os.getpid())); time.sleep(30)'"
    result = runner.run(f"{spawn} & while [ ! -s pid ]; do sleep 0.05; done; echo done", cwd=tmp_path,
                        timeout_s=20, max_output_bytes=1000)
    assert result.exit_code == 0 and result.stdout.text == "done\n"
    pid = int((tmp_path / "pid").read_text())
    for _ in range(100):  # SIGKILL is asynchronous; give it up to 2 s
        if not _alive(pid):
            break
        time.sleep(0.02)
    assert not _alive(pid)


def test_pipelines_fail_when_any_stage_fails(tmp_path: Path) -> None:
    runner = SubprocessRunner({"PATH": "/usr/bin:/bin"})
    result = runner.run(f"{PY} -c 'import sys; print(\"1 failed\"); sys.exit(3)' | tail -1", cwd=tmp_path,
                        timeout_s=20, max_output_bytes=1000)
    assert result.stdout.text == "1 failed\n"
    assert result.exit_code == 3  # tail succeeded, but the test command's failure is what counts


def test_stdin_is_empty_and_secrets_are_scrubbed(tmp_path: Path) -> None:
    environ = {"PATH": "/usr/bin:/bin", "AGENT_API_KEY": "sk-secret", "OPENAI_API_KEY": "sk-2", "KEEP": "1"}
    code = "import os, sys; print(repr(sys.stdin.read())); print(sorted(k for k in os.environ if 'KEY' in k or k == 'KEEP'))"
    result = run(tmp_path, code, environ=environ)
    assert result.stdout.text == "''\n['KEEP']\n"
    assert child_environment({"X": "1"})["GIT_TERMINAL_PROMPT"] == "0"


def test_the_first_failing_command_stops_the_script(tmp_path: Path) -> None:
    runner = SubprocessRunner({"PATH": "/usr/bin:/bin"})
    result = runner.run(f"{PY} -c 'import sys; sys.exit(4)'; echo \"exit=$?\"", cwd=tmp_path,
                        timeout_s=20, max_output_bytes=1000)
    assert result.exit_code == 4 and result.stdout.text == ""  # the echo never ran, so it cannot launder the 4
