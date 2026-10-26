"""Host seams: time and processes.

Both are injected into the loop and tools, so tests never depend on the real
clock or on what is installed on the machine.

Decisions owned here:
  * Commands run under ``bash -e -o pipefail -c`` (``/bin/sh -e -c`` only where
    bash is missing), the settings GitHub Actions uses: ``pytest | tail -50``
    reports pytest's failure instead of tail's success, and ``pytest; echo
    done`` stops at the failing test. The verification gate trusts exit codes,
    so a command line must not launder one. (``tests || true`` still can; the
    policy flags it and the gate does not accept it as a passing check.)
  * Each command gets its own session (process group). When the command exits
    *or* times out, the whole group is killed, so a stray ``server &`` cannot
    outlive the call or hold the pipes open.
  * Output is captured with bounded memory: the first and last halves of the
    byte cap are kept per stream, because a failing test prints its context at
    the top and its verdict at the bottom.
  * stdin is /dev/null and pagers/credential prompts are disabled, so an
    interactive command fails fast instead of hanging until the timeout.
  * The harness's own provider credentials are removed from the child
    environment: a prompt-injected ``env | curl ...`` cannot exfiltrate them.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Protocol

SECRET_ENV_VARS = ("AGENT_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
_NONINTERACTIVE_ENV = {"GIT_TERMINAL_PROMPT": "0", "PAGER": "cat", "GIT_PAGER": "cat"}


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def now_iso(self) -> str: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def now_iso(self) -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class CapturedStream:
    text: str
    total_bytes: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int | None  # None when timed out; negative N when killed by signal N
    stdout: CapturedStream
    stderr: CapturedStream
    timed_out: bool
    duration_s: float


class CommandRunner(Protocol):
    def run(self, command: str, *, cwd: Path, timeout_s: float, max_output_bytes: int) -> ProcessResult: ...


def child_environment(base: Mapping[str, str], scrub: Iterable[str] = SECRET_ENV_VARS) -> dict[str, str]:
    removed = set(scrub)
    env = {key: value for key, value in base.items() if key not in removed}
    env.update(_NONINTERACTIVE_ENV)
    return env


class _HeadTailBuffer:
    """Keeps the first and last ``limit // 2`` bytes of a stream."""

    def __init__(self, limit: int) -> None:
        self._head_limit = limit - limit // 2
        self._tail_limit = limit // 2
        self._head = bytearray()
        self._tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self._head_limit - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk and self._tail_limit > 0:
            self._tail += chunk
            excess = len(self._tail) - self._tail_limit
            if excess > 0:
                del self._tail[:excess]

    def result(self) -> CapturedStream:
        kept = len(self._head) + len(self._tail)
        head = bytes(self._head).decode("utf-8", errors="replace")
        if self.total <= kept:
            return CapturedStream(text=head + bytes(self._tail).decode("utf-8", errors="replace"),
                                  total_bytes=self.total, truncated=False)
        omitted = self.total - kept
        tail = bytes(self._tail).decode("utf-8", errors="replace")
        marker = f"\n[... {omitted} bytes omitted ...]\n"
        return CapturedStream(text=head + marker + tail, total_bytes=self.total, truncated=True)


def _pump(stream: IO[bytes], buffer: _HeadTailBuffer) -> None:
    fd = stream.fileno()
    try:
        while chunk := os.read(fd, 65536):
            buffer.feed(chunk)
    except OSError:
        pass
    finally:
        stream.close()


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def shell_argv(command: str) -> list[str]:
    bash = shutil.which("bash")
    return [bash, "-e", "-o", "pipefail", "-c", command] if bash else ["/bin/sh", "-e", "-c", command]


class SubprocessRunner:
    def __init__(self, environ: Mapping[str, str] | None = None, scrub: Iterable[str] = SECRET_ENV_VARS) -> None:
        self._env = child_environment(os.environ if environ is None else environ, scrub)

    def run(self, command: str, *, cwd: Path, timeout_s: float, max_output_bytes: int) -> ProcessResult:
        started = time.monotonic()
        proc = subprocess.Popen(
            shell_argv(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            start_new_session=True,
        )
        assert proc.stdout is not None and proc.stderr is not None
        out, err = _HeadTailBuffer(max_output_bytes), _HeadTailBuffer(max_output_bytes)
        pumps = [
            threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True),
        ]
        for pump in pumps:
            pump.start()
        timed_out = False
        try:
            proc.wait(timeout=max(timeout_s, 0.0))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _kill_group(proc.pid)
            proc.wait()
        for pump in pumps:
            pump.join(timeout=2.0)  # a process that escaped the group can hold a pipe open
        return ProcessResult(
            exit_code=None if timed_out else proc.returncode,
            stdout=out.result(),
            stderr=err.result(),
            timed_out=timed_out,
            duration_s=time.monotonic() - started,
        )
