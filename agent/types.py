"""Shared vocabulary. Every value that crosses a module boundary is defined here.

Nothing in this module makes a decision; it exists so that the modules which do
(providers, tools, ledger, loop) agree on one set of types.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Literal

type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]

Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation requested by the model.

    ``arguments`` is the raw JSON text exactly as the model produced it. Parsing
    belongs to the loop, so unparseable arguments become a malformed-output
    event instead of an exception inside a provider adapter.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""  # set only when role == "tool"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """What a provider returns: text, tool calls, or both.

    ``usage`` is None when the backend did not report token counts; the loop
    then estimates them and marks the event as estimated.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None
    finish_reason: str = ""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool as the model sees it. ``parameters`` is a JSON Schema document."""

    name: str
    description: str
    parameters: JSONObject


# --------------------------------------------------------------------------- #
# Tool results and their effects on the workspace
# --------------------------------------------------------------------------- #


class CommandKind(enum.StrEnum):
    """Approval classes for shell commands.

    READ_ONLY means *known* to have no side effects. Anything not known to be
    inert is MUTATING. DESTRUCTIVE means it can discard work irrecoverably.
    """

    READ_ONLY = "read_only"
    MUTATING = "mutating"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class UntrustedText:
    """Content the harness did not author: file bytes, command output, paths,
    a subagent's prose. It reaches the model only inside delimiters."""

    source: str
    text: str


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str  # workspace-relative, POSIX separators
    action: Literal["created", "overwritten", "edited"]
    detail: str
    sha256: str
    lines: int


@dataclass(frozen=True, slots=True)
class CommandRun:
    command: str
    cwd: str
    kind: CommandKind
    exit_code: int | None  # None when the command timed out
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    duration_s: float
    masks_exit_status: bool = False  # `tests || true`: exit 0 does not mean the checks passed


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    command: str
    kind: CommandKind
    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ReadSpan:
    """Lines ``first``..``last`` (1-based, inclusive) of a file that had ``total`` lines."""

    path: str
    first: int
    last: int
    total: int
    truncated: bool = False  # some lines were cut, so this read does not count as seeing them


@dataclass(frozen=True, slots=True)
class Effects:
    """Facts a tool reports about what it did. The loop records them in the
    ledger; tools never write to the ledger themselves."""

    reads: tuple[ReadSpan, ...] = ()
    changes: tuple[FileChange, ...] = ()
    command: CommandRun | None = None
    approval: ApprovalRecord | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Outcome of one tool call.

    ``message`` is harness-authored and trusted. Anything derived from file
    contents or process output goes in ``data`` so it is rendered delimited.
    """

    ok: bool
    message: str
    data: tuple[UntrustedText, ...] = ()
    error_code: str = ""
    effects: Effects = field(default_factory=Effects)

    @staticmethod
    def error(code: str, message: str, *data: UntrustedText) -> ToolResult:
        return ToolResult(ok=False, message=message, data=tuple(data), error_code=code)


# --------------------------------------------------------------------------- #
# Run configuration and outcome
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Limits:
    max_iterations: int = 40  # model calls: run-wide for the root agent, per subagent for a child
    max_total_tokens: int = 1_500_000  # prompt + completion, summed over calls
    max_wall_seconds: float = 1800.0
    context_window_tokens: int = 100_000
    compact_at_fraction: float = 0.75  # compact when the next prompt would exceed this share of the window
    compact_to_fraction: float = 0.40  # aim for a prompt this size after compaction (overhead included)
    max_consecutive_malformed: int = 3
    loop_repeat_threshold: int = 3  # identical consecutive calls before intervention
    replan_after_failed_attempts: int = 2
    command_timeout_seconds: float = 120.0
    max_command_timeout_seconds: float = 600.0
    max_output_bytes: int = 16_000  # per stream, split between head and tail
    delegate_max_iterations: int = 15


OutcomeStatus = Literal["done", "blocked", "budget_exhausted"]
OutcomeCause = Literal[
    "reported",  # done/blocked as reported by the agent via `finish`
    "unverified",  # `done` downgraded by the verification gate
    "malformed_output",
    "provider_error",
    "loop_detected",
    "interrupted",
    "internal_error",
    "iterations",
    "tokens",
    "wall_clock",
    "context",
]


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """A successful command the harness itself observed after the last edit."""

    step: int
    command: str
    exit_code: int
    output_tail: str


@dataclass(frozen=True, slots=True)
class Outcome:
    status: OutcomeStatus
    cause: OutcomeCause
    message: str  # summary when done, reason when blocked, which cap when exhausted
    evidence: tuple[VerificationEvidence, ...] = ()
    claimed_evidence: tuple[str, ...] = ()  # what the model said; not trusted
    files_modified: tuple[str, ...] = ()
    iterations: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
