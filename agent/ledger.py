"""What the run actually did to the workspace, as observed by the harness.

The model's account of its work is a claim; the ledger is the record. It is
shared between an agent and its subagents, because verification is a property
of the workspace, not of whichever agent happened to make an edit.

Decisions owned here:
  * A *modification* is a successful write_file or edit_file. Edits made
    through run_command (``sed -i``, a script) are not seen; the system prompt
    routes all edits through the file tools and DESIGN.md names the gap.
  * A *check* is a run_command that is not classified READ_ONLY. ``ls``,
    ``cat`` or ``echo ok`` demonstrate nothing about a change, so they are
    neither evidence nor counter-evidence.
  * The gate: if anything was modified, ``done`` needs at least one passing
    check after the last modification, and every check that *failed* after it
    must have been re-run (same command, same directory) and passed. A failing
    test followed by an unrelated ``mkdir`` is not verification, and a command
    whose exit status is masked (``tests || true``) cannot pass, only fail.
  * Overwriting a file requires having seen all of it: the ledger merges the
    line ranges read_file returned (a page with cut lines does not count), so
    reading line 1 of a 3,000-line file does not license replacing it. Having
    *written* the whole file also counts; having edited part of it does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.types import CommandKind, CommandRun, Effects, FileChange, ReadSpan, VerificationEvidence
from agent.untrusted import inline

_EVIDENCE_TAIL_CHARS = 600


@dataclass(frozen=True, slots=True)
class Modification:
    step: int
    agent: str
    change: FileChange


@dataclass(frozen=True, slots=True)
class CommandEntry:
    step: int
    agent: str
    run: CommandRun

    @property
    def succeeded(self) -> bool:
        return self.run.exit_code == 0 and not self.run.timed_out

    @property
    def is_check(self) -> bool:
        return self.run.kind is not CommandKind.READ_ONLY

    @property
    def passes(self) -> bool:
        return self.succeeded and not self.run.masks_exit_status

    def same_check(self, other: CommandEntry) -> bool:
        return self.run.command == other.run.command and self.run.cwd == other.run.cwd

    def status(self) -> str:
        return "timed out" if self.run.timed_out else f"exit {self.run.exit_code}"

    def output_tail(self, limit: int = _EVIDENCE_TAIL_CHARS) -> str:
        combined = "\n".join(part for part in (self.run.stdout_tail, self.run.stderr_tail) if part.strip())
        return combined[-limit:]


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    explanation: str
    evidence: tuple[VerificationEvidence, ...]


def _merge(spans: list[tuple[int, int]], first: int, last: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted([*spans, (first, last)]):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class Ledger:
    def __init__(self) -> None:
        self.step = 0
        self.modifications: list[Modification] = []
        self.commands: list[CommandEntry] = []
        self._read_spans: dict[str, list[tuple[int, int]]] = {}
        self._read_totals: dict[str, int] = {}

    def record(self, agent: str, effects: Effects) -> int:
        """Record one tool execution (even one with no effects) and return its step."""
        self.step += 1
        for span in effects.reads:
            self._record_read(span)
        for change in effects.changes:
            self.modifications.append(Modification(self.step, agent, change))
        if effects.command is not None:
            self.commands.append(CommandEntry(self.step, agent, effects.command))
        return self.step

    def _record_read(self, span: ReadSpan) -> None:
        if self._read_totals.get(span.path) != span.total:
            self._read_spans[span.path] = []  # the file changed size: earlier coverage is stale
        self._read_totals[span.path] = span.total
        spans = self._read_spans.setdefault(span.path, [])
        if span.total > 0 and not span.truncated:
            self._read_spans[span.path] = _merge(spans, span.first, span.last)

    def fully_read(self, path: str) -> bool:
        total = self._read_totals.get(path)
        if total is None:
            return False
        return total == 0 or self._read_spans.get(path, [])[:1] == [(1, total)]

    def may_overwrite(self, path: str) -> bool:
        """True if the agent has seen the whole file, or wrote all of it itself, during this run."""
        return self.fully_read(path) or any(
            m.change.path == path and m.change.action in ("created", "overwritten") for m in self.modifications)

    def last_modification(self, by: str | None = None) -> Modification | None:
        """The latest modification, optionally only those made by agent ``by``."""
        for mod in reversed(self.modifications):
            if by is None or mod.agent == by:
                return mod
        return None

    def modifications_between(self, after_step: int, before_step: int) -> bool:
        return any(after_step < m.step < before_step for m in self.modifications)

    def files_modified(self, by: str | None = None) -> tuple[str, ...]:
        return tuple(dict.fromkeys(m.change.path for m in self.modifications if by is None or m.agent == by))

    def file_states(self) -> list[Modification]:
        """The latest modification of each file, in first-touched order."""
        latest: dict[str, Modification] = {}
        for mod in self.modifications:
            latest[mod.change.path] = mod
        return [latest[path] for path in self.files_modified()]

    def files_read_only(self) -> list[str]:
        modified = set(self.files_modified())
        return [path for path in self._read_totals if path not in modified]

    def unresolved_failures(self, limit: int = 5) -> list[CommandEntry]:
        """Most recent failing checks whose command has not passed since."""
        unresolved: list[CommandEntry] = []
        seen: set[tuple[str, str]] = set()
        for entry in reversed(self.commands):
            key = (entry.run.command, entry.run.cwd)
            if key in seen or not entry.is_check:
                continue
            seen.add(key)
            if not entry.succeeded:
                unresolved.append(entry)
            if len(unresolved) >= limit:
                break
        return unresolved

    def gate(self, by: str | None = None) -> GateResult:
        """Check verification after the last modification.

        ``by`` scopes the check to one agent's edits: a subagent answers for
        its own changes, while the root agent answers for the whole workspace.
        Checks by any agent count, as long as they come later.
        """
        last = self.last_modification(by)
        if last is None:
            return GateResult(True, "No files were modified, so there was nothing to verify.", ())
        after = [c for c in self.commands if c.step > last.step]
        checks = [c for c in after if c.is_check]
        passed = [c for c in checks if c.passes]
        failed = [c for c in checks if not c.succeeded and not any(p.step > c.step and p.same_check(c) for p in passed)]
        masked = [c for c in checks if c.succeeded and c.run.masks_exit_status]
        if passed and not failed:
            evidence = tuple(VerificationEvidence(c.step, c.run.command, 0, c.output_tail()) for c in passed)
            return GateResult(True, f"Verified by {len(passed)} passing check(s) after the last modification.",
                              evidence)
        explanation = (
            f"There is no passing verification after the last file modification "
            f"({inline(last.change.path)}, step {last.step})."
        )
        if failed:
            worst = failed[-1]
            explanation += (
                f" `{inline(worst.run.command)}` failed after it (step {worst.step}, {worst.status()}) and was "
                "not re-run successfully; a failing check means the change is not verified, whatever else passed."
            )
        elif masked:
            explanation += (f" `{inline(masked[-1].run.command)}` exited 0, but its exit status is masked (for example "
                            "by `|| true`), so it does not show that the checks passed.")
        elif after:
            explanation += " The commands run after it were read-only, which does not demonstrate a change."
        else:
            explanation += " No command was run after it."
        return GateResult(False, explanation, ())

    def verification_status(self, by: str | None = None) -> str:
        last = self.last_modification(by)
        if last is None:
            return "No files modified yet."
        gate = self.gate(by)
        if gate.passed:
            latest = gate.evidence[-1]
            return (f"VERIFIED: `{inline(latest.command)}` passed at step {latest.step}, after the last "
                    f"modification (step {last.step}).")
        return f"NOT VERIFIED: {gate.explanation}"
