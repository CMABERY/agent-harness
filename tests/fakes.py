"""Test doubles for the host seams, and a helper that wires an agent for a test."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agent.context import CharTokenCounter, TokenCounter
from agent.events import EventLog, EventRecord
from agent.host import CapturedStream, ProcessResult
from agent.loop import Agent, Runtime
from agent.policy import ApprovalPolicy, DefaultPolicy
from agent.providers import ScriptedProvider, ScriptStep
from agent.types import Limits, Message
from agent.workspace import Workspace


def payload(record: EventRecord) -> Any:
    """An event's data as plain JSON, for digging into nested fields in assertions."""
    return record.data


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._start = start
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def now_iso(self) -> str:
        moment = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self.now - self._start)
        return moment.isoformat(timespec="milliseconds")

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass(frozen=True)
class Canned:
    exit_code: int | None = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.5


@dataclass
class FakeRunner:
    """Returns canned results by command string and advances the fake clock.

    A command mapped to a list yields its items in order (fail, then pass);
    ``handler`` can compute a result from the command and cwd instead.
    """

    clock: FakeClock
    results: dict[str, Canned | list[Canned]] = field(default_factory=dict)
    handler: Callable[[str, Path], Canned] | None = None
    calls: list[tuple[str, Path, float]] = field(default_factory=list)

    def run(self, command: str, *, cwd: Path, timeout_s: float, max_output_bytes: int) -> ProcessResult:
        self.calls.append((command, cwd, timeout_s))
        canned: Canned
        if self.handler is not None:
            canned = self.handler(command, cwd)
        else:
            entry = self.results.get(command, Canned())
            canned = entry.pop(0) if isinstance(entry, list) else entry
        self.clock.advance(canned.duration_s)
        return ProcessResult(
            exit_code=None if canned.timed_out else canned.exit_code,
            stdout=CapturedStream(canned.stdout, len(canned.stdout.encode()), False),
            stderr=CapturedStream(canned.stderr, len(canned.stderr.encode()), False),
            timed_out=canned.timed_out,
            duration_s=canned.duration_s,
        )


@dataclass
class Rig:
    agent: Agent
    provider: ScriptedProvider
    runner: FakeRunner
    clock: FakeClock
    events: EventLog
    workspace: Workspace

    def sent(self, request: int = -1) -> list[Message]:
        """The conversation exactly as the provider received it on one call."""
        return list(self.provider.requests[request].messages)

    def tool_messages(self, request: int = -1) -> list[str]:
        return [m.content for m in self.sent(request) if m.role == "tool"]

    def kinds(self) -> list[str]:
        return [r.kind for r in self.events.records]


def make_rig(
    root: Path,
    script: Sequence[ScriptStep],
    *,
    limits: Limits | None = None,
    results: dict[str, Canned | list[Canned]] | None = None,
    handler: Callable[[str, Path], Canned] | None = None,
    policy: ApprovalPolicy | None = None,
    log_path: Path | None = None,
    system_prompt: str | None = "You are a test agent.",
    counter: TokenCounter | None = None,
    clock: FakeClock | None = None,
) -> Rig:
    clock = clock or FakeClock()
    runner = FakeRunner(clock, results or {}, handler)
    workspace = Workspace(root)
    events = EventLog(log_path, clock)
    provider = ScriptedProvider(script)
    runtime = Runtime(
        clock=clock,
        runner=runner,
        policy=policy or DefaultPolicy(workspace.root),
        events=events,
        counter=counter or CharTokenCounter(),
    )
    agent = Agent(provider, workspace, runtime, limits=limits, system_prompt=system_prompt)
    return Rig(agent, provider, runner, clock, events, workspace)
