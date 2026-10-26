"""Token accounting and context compaction.

Decisions owned here:
  * Counting sits behind ``TokenCounter``; the default is characters / 4 plus
    a per-message overhead. When the provider reports real prompt tokens, the
    meter keeps the ratio between the real count and the character count, and
    every estimate (before or after compaction, for the trigger and for the
    "did it shrink" check alike) is scaled by it. Mixing a calibrated number
    with a raw one would let a compaction that grew the prompt look smaller.
  * Compaction is deterministic and written by the harness from its own
    records, not by asking the model to summarize. An LLM summary of turns
    that contain untrusted file content would turn that content into
    harness-authored prose, which is exactly the laundering the untrusted-data
    delimiters exist to prevent. A deterministic summary also costs no model
    call, is testable offline, and cannot drop the facts the verification
    gate and replan rule depend on. The cost is the model's free-form
    reasoning in old turns; the plan's ``notes`` field is the channel for
    anything the model wants to keep, and it survives verbatim.
  * Size: the target after compaction (``compact_to_fraction`` of the window)
    covers everything sent, including the system prompt and tool specs, so the
    verbatim tail gets what is left after that fixed overhead and the summary.
    A compaction that would not shrink the prompt is not applied.
  * Preserved: the system prompt and the original task message (verbatim);
    the latest plan (verbatim); every modified file with its last known state;
    files read; verification status; unresolved failures with their output
    tail (still delimited as untrusted); a one-line-per-action log of earlier
    work so it is not repeated; and the most recent turns verbatim.
  * The verbatim tail starts on a turn boundary, so an assistant message is
    never separated from its tool results.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from agent.ledger import Ledger
from agent.types import Message, ToolSpec, UntrustedText
from agent.untrusted import printable, wrap

_ACTION_LOG_LIMIT = 40


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Sequence[Message]) -> int: ...

    def count_tools(self, tools: Sequence[ToolSpec]) -> int: ...


class CharTokenCounter:
    PER_MESSAGE = 4

    def count_text(self, text: str) -> int:
        return (len(text) + 3) // 4

    def count_messages(self, messages: Sequence[Message]) -> int:
        total = 0
        for message in messages:
            total += self.PER_MESSAGE + self.count_text(message.content)
            for call in message.tool_calls:
                total += self.PER_MESSAGE + self.count_text(call.name) + self.count_text(call.arguments)
        return total

    def count_tools(self, tools: Sequence[ToolSpec]) -> int:
        return sum(
            self.PER_MESSAGE + self.count_text(t.name + t.description + json.dumps(t.parameters)) for t in tools
        )


class ContextMeter:
    """Estimates prompt size in the provider's tokens: the character-based count
    scaled by the latest observed ratio of reported to counted tokens."""

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter
        self.ratio = 1.0

    def raw(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> int:
        return self._counter.count_messages(messages) + self._counter.count_tools(tools)

    def observe(self, raw_prompt_tokens: int, reported_prompt_tokens: int | None) -> None:
        if reported_prompt_tokens is not None and raw_prompt_tokens > 0:
            self.ratio = min(max(reported_prompt_tokens / raw_prompt_tokens, 0.25), 4.0)

    def estimate(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> int:
        return math.ceil(self.raw(messages, tools) * self.ratio)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    step: int
    tool: str
    arguments: str
    result: str


def abbreviate(text: str, limit: int = 120) -> str:
    flat = printable(text)
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def build_summary(
    removed: int, plan_text: str, ledger: Ledger, actions: Sequence[ActionRecord], scope: str | None = None
) -> str:
    sections = [
        f"[harness] Context compacted: {removed} earlier message(s) were replaced by this summary. It was "
        "written by the harness from its own records, not by a model. The original task is still the first "
        "user message. File contents are not included here: re-read a file before editing it.",
        "== Your plan (as you last wrote it) ==\n" + plan_text,
    ]
    states = ledger.file_states()
    if states:
        lines = [
            f"- {printable(m.change.path)}: {m.change.action} at step {m.step} ({m.change.detail}); "
            f"now {m.change.lines} lines, sha256 {m.change.sha256}"
            for m in states
        ]
        sections.append("== Files modified this run (last known state) ==\n" + "\n".join(lines))
    else:
        sections.append("== Files modified this run ==\nnone")
    read_only = ledger.files_read_only()
    if read_only:
        sections.append("== Files read but not modified ==\n" + ", ".join(printable(p) for p in read_only))
    sections.append("== Verification ==\n" + ledger.verification_status(scope))
    failures = ledger.unresolved_failures()
    if failures:
        entries = []
        for entry in failures:
            code = "timed out" if entry.run.timed_out else f"exited {entry.run.exit_code}"
            tail = entry.output_tail(800)
            block = wrap(UntrustedText(f"command output, step {entry.step}", tail)) if tail else "(no output)"
            entries.append(f"- step {entry.step}: `{abbreviate(entry.run.command, 200)}` {code}. Output tail:\n{block}")
        sections.append("== Unresolved failures (most recent first) ==\n" + "\n".join(entries))
    if actions:
        shown = actions[-_ACTION_LOG_LIMIT:]
        header = f"== Earlier actions, oldest first ({len(shown)} of {len(actions)}) =="
        sections.append(header + "\n" + "\n".join(
            f"{a.step:>4}. {a.tool} {a.arguments} -> {a.result}" for a in shown
        ))
    return "\n\n".join(sections)


def select_tail(messages: Sequence[Message], keep_tokens: int, counter: TokenCounter) -> int | None:
    """Index into ``messages`` where the verbatim tail starts, or None if nothing can be removed.

    ``messages[0:2]`` (system prompt and task) are never candidates. The tail
    starts at a non-tool message so tool results stay with their call.
    """
    body_start = 2
    sizes = [counter.count_messages([m]) for m in messages]
    suffix = [0] * (len(messages) + 1)
    for index in range(len(messages) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + sizes[index]
    boundaries = [i for i in range(body_start + 1, len(messages)) if messages[i].role != "tool"]
    for index in boundaries:
        if suffix[index] <= keep_tokens:
            return index
    return boundaries[-1] if boundaries else None


def compact(messages: Sequence[Message], tail_start: int, task: str, summary: str) -> list[Message]:
    """System prompt, then one user message holding the original task and the
    summary, then the verbatim tail. Task and summary share a message (and a
    user message opening the tail joins them) because some chat templates
    reject two user messages in a row."""
    head = f"{task}\n\n{summary}"
    tail = list(messages[tail_start:])
    if tail and tail[0].role == "user":
        head += f"\n\n{tail.pop(0).content}"
    return [messages[0], Message("user", head), *tail]
