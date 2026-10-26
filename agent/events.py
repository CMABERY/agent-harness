"""Append-only JSONL event log.

Decisions owned here:
  * One JSON object per line, flushed as it is written, so a run that crashes
    or is killed still leaves a complete record up to that point.
  * Every message that enters a conversation is logged (``message_added``,
    with its index: an index equal to the current length appends, a smaller
    one replaces from that position, which is how a merged user message is
    recorded), and compaction logs the full replacement conversation.
    Replaying those two event kinds reproduces exactly what the model saw on
    every call; see ``reconstruct_conversation``.
  * Lines are ASCII-escaped JSON, so nothing a file name or a command prints
    (undecodable bytes included) can make a write fail.
  * A failing log write never ends the run: it is recorded on the log object
    and reported by the CLI. Losing observability is bad; losing the user's
    work because the disk filled up is worse.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TextIO

from agent.host import Clock
from agent.types import (
    Effects,
    JSONObject,
    JSONValue,
    Limits,
    Message,
    ModelResponse,
    Outcome,
    Role,
    TokenUsage,
    ToolCall,
)


@dataclass(frozen=True, slots=True)
class Event:
    kind: ClassVar[str] = "event"


@dataclass(frozen=True, slots=True)
class RunStarted(Event):
    kind: ClassVar[str] = "run_started"
    task: str
    workspace: str
    depth: int
    tools: tuple[str, ...]
    limits: Limits


@dataclass(frozen=True, slots=True)
class MessageAdded(Event):
    kind: ClassVar[str] = "message_added"
    index: int
    message: Message


@dataclass(frozen=True, slots=True)
class ModelCalled(Event):
    kind: ClassVar[str] = "model_called"
    iteration: int
    duration_s: float
    usage: TokenUsage
    usage_estimated: bool
    prompt_estimate: int
    finish_reason: str
    tool_call_count: int


@dataclass(frozen=True, slots=True)
class MalformedOutput(Event):
    kind: ClassVar[str] = "malformed_output"
    iteration: int
    consecutive: int
    problem: str
    response: ModelResponse  # discarded from the conversation, kept here


@dataclass(frozen=True, slots=True)
class ToolExecuted(Event):
    kind: ClassVar[str] = "tool_executed"
    call_id: str
    name: str
    arguments: str
    ok: bool
    error_code: str
    duration_s: float
    effects: Effects


@dataclass(frozen=True, slots=True)
class LoopDetected(Event):
    kind: ClassVar[str] = "loop_detected"
    name: str
    arguments: str
    repeats: int
    terminated: bool


@dataclass(frozen=True, slots=True)
class ContextCompacted(Event):
    kind: ClassVar[str] = "context_compacted"
    tokens_before: int
    tokens_after: int
    removed_messages: int
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class PlanUpdated(Event):
    kind: ClassVar[str] = "plan_updated"
    version: int
    rendered: str


@dataclass(frozen=True, slots=True)
class ReplanRequired(Event):
    kind: ClassVar[str] = "replan_required"
    step: str
    failed_attempts: int


@dataclass(frozen=True, slots=True)
class VerificationDowngraded(Event):
    kind: ClassVar[str] = "verification_downgraded"
    reason: str
    claimed_summary: str


@dataclass(frozen=True, slots=True)
class InternalError(Event):
    kind: ClassVar[str] = "internal_error"
    where: str
    error: str
    traceback: str


@dataclass(frozen=True, slots=True)
class RunFinished(Event):
    kind: ClassVar[str] = "run_finished"
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class EventRecord:
    seq: int
    ts: str
    elapsed_s: float
    agent: str
    kind: str
    data: JSONObject


def _to_json(event: Event) -> JSONObject:
    data: JSONObject = json.loads(json.dumps(dataclasses.asdict(event), default=str))
    return data


class EventLog:
    def __init__(self, path: Path | None, clock: Clock) -> None:
        self.path = path
        self.records: list[EventRecord] = []
        self.write_error: str = ""
        self._clock = clock
        self._start = clock.monotonic()
        self._handle: TextIO | None = None
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = path.open("a", encoding="utf-8")
            except OSError as exc:
                self.write_error = f"cannot open event log {path}: {exc}"

    def emit(self, agent: str, event: Event) -> EventRecord:
        record = EventRecord(
            seq=len(self.records) + 1,
            ts=self._clock.now_iso(),
            elapsed_s=round(self._clock.monotonic() - self._start, 6),
            agent=agent,
            kind=event.kind,
            data=_to_json(event),
        )
        self.records.append(record)
        if self._handle is not None:
            try:
                self._handle.write(json.dumps(dataclasses.asdict(record), ensure_ascii=True, default=str) + "\n")
                self._handle.flush()
            except (OSError, ValueError) as exc:
                self.write_error = f"event log write failed: {exc}"
                self._handle = None
        return record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# --------------------------------------------------------------------------- #
# Reading a log back
# --------------------------------------------------------------------------- #


def read_log(path: Path) -> list[EventRecord]:
    records: list[EventRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(
            EventRecord(
                seq=int(raw["seq"]),
                ts=str(raw["ts"]),
                elapsed_s=float(raw["elapsed_s"]),
                agent=str(raw["agent"]),
                kind=str(raw["kind"]),
                data=raw["data"],
            )
        )
    return records


def _object(value: JSONValue) -> JSONObject:
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in the event log, got {type(value).__name__}")
    return value


_ROLES: dict[str, Role] = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"}


def message_from_json(value: JSONValue) -> Message:
    obj = _object(value)
    role = _ROLES.get(str(obj.get("role")))
    if role is None:
        raise ValueError(f"bad role in event log: {obj.get('role')!r}")
    raw_calls = obj.get("tool_calls", [])
    calls = tuple(
        ToolCall(id=str(c["id"]), name=str(c["name"]), arguments=str(c["arguments"]))
        for c in (raw_calls if isinstance(raw_calls, list) else [])
        if isinstance(c, dict)
    )
    return Message(
        role=role,
        content=str(obj.get("content", "")),
        tool_calls=calls,
        tool_call_id=str(obj.get("tool_call_id", "")),
    )


def reconstruct_conversation(records: list[EventRecord], agent: str = "main") -> list[Message]:
    """Replay the log into the conversation as it stood after the last event."""
    messages: list[Message] = []
    for record in records:
        if record.agent != agent:
            continue
        if record.kind == ContextCompacted.kind:
            raw = record.data.get("messages", [])
            messages = [message_from_json(m) for m in (raw if isinstance(raw, list) else [])]
        elif record.kind == MessageAdded.kind:
            index = record.data.get("index")
            position = index if isinstance(index, int) else len(messages)
            messages[position:] = [message_from_json(record.data["message"])]
    return messages
