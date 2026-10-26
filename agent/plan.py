"""The agent's plan, and the harness rule that forces a replan.

Decisions owned here:
  * The plan is model-authored state that the harness stores, renders and
    carries through compaction verbatim. ``update_plan`` replaces it whole,
    which keeps the tool idempotent and removes step-id bookkeeping.
  * An *attempt* is the window that opens with a modification and closes at
    the next one. It fails if any check (a command that is not read-only) fails
    inside it, and it is charged once to the step in progress. A passing
    ``pip install`` in the same window does not rescue a failing test.
    Reproducing a failure before the first edit, and re-running a failing check
    without editing, do not open a window, so the rule punishes repeating a
    fix, not investigating.
  * After ``threshold`` failed attempts on one step, mutating tools are
    refused until the model submits a plan whose step titles differ from the
    current ones. Flipping a status is not a new approach. Reading, searching
    and running commands stay available so the model can re-diagnose. Without
    any plan, the failures are charged to "(no plan)" and the same rule makes
    the model write one: a task that has failed twice is not trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.ledger import Ledger
from agent.schema import ToolArgs
from agent.types import CommandKind, Effects, ToolResult

StepStatus = Literal["pending", "in_progress", "done", "failed"]
_STATUSES: dict[str, StepStatus] = {
    "pending": "pending", "in_progress": "in_progress", "done": "done", "failed": "failed",
}
_MARKS = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]", "failed": "[!]"}
NO_PLAN = "(no plan)"


@dataclass(frozen=True, slots=True)
class PlanStep:
    title: str
    status: StepStatus


@dataclass(frozen=True, slots=True)
class Plan:
    version: int
    acceptance_criteria: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    notes: str

    def current_step(self) -> str:
        for wanted in ("in_progress", "pending"):
            for step in self.steps:
                if step.status == wanted:
                    return step.title
        return "(no open step)"

    def render(self) -> str:
        lines = [f"Plan v{self.version}", "Acceptance criteria:"]
        lines.extend(f"  - {c}" for c in self.acceptance_criteria)
        lines.append("Steps:")
        lines.extend(f"  {_MARKS[s.status]} {s.title}" for s in self.steps)
        if self.notes.strip():
            lines.append(f"Notes: {self.notes.strip()}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReplanTrigger:
    step: str
    failed_attempts: int
    note: str


class PlanTracker:
    def __init__(self, threshold: int, start_step: int = 0) -> None:
        """``start_step`` is the ledger step when this agent started: edits made
        before it (by a parent, for a subagent) are not this agent's attempts."""
        self.plan: Plan | None = None
        self.replan_reason = ""
        self._threshold = threshold
        self._failures: dict[str, int] = {}
        self._last_observed_step = start_step
        self._window_open = False  # a modification happened and no failure was charged since

    def update(self, args: ToolArgs) -> ToolResult:
        steps = tuple(PlanStep(s.text("title").strip(), _STATUSES[s.text("status")]) for s in args.object_list("steps"))
        if sum(1 for s in steps if s.status == "in_progress") > 1:
            return ToolResult.error("invalid_plan", "At most one step may be in_progress; the plan was not changed.")
        if self.replan_reason and self.plan is not None and [s.title for s in steps] == [
            s.title for s in self.plan.steps
        ]:
            return ToolResult.error(
                "plan_unchanged",
                f"A revised plan is required ({self.replan_reason}), but its steps have the same titles as the "
                "current plan; changing a status is not a new approach. Replace, split or reorder the failing step.",
            )
        version = self.plan.version + 1 if self.plan else 1
        self.plan = Plan(version, tuple(args.text_list("acceptance_criteria")), steps, args.text("notes"))
        counts = {status: sum(1 for s in steps if s.status == status) for status in _MARKS}
        message = (
            f"Plan v{version} recorded: {len(steps)} step(s) — {counts['done']} done, "
            f"{counts['in_progress']} in progress, {counts['pending']} pending, {counts['failed']} failed."
        )
        if self.replan_reason:
            self.replan_reason = ""
            self._failures.clear()
            message += " The replan requirement is cleared; edits are allowed again."
        return ToolResult(True, message)

    def observe(self, step: int, effects: Effects, ledger: Ledger) -> ReplanTrigger | None:
        """Charge failed attempts after each tool execution; report when a replan becomes required.

        Called once per tool execution of this agent, with the ledger step it
        was recorded at. Modifications made by a subagent during a delegate
        call fall between two observed steps and open a window too.
        """
        if ledger.modifications_between(self._last_observed_step, step + 1):
            self._window_open = True
        self._last_observed_step = step
        command = effects.command
        if command is None or command.kind is CommandKind.READ_ONLY:
            return None
        if command.exit_code == 0 and not command.timed_out:
            return None
        if not self._window_open:
            return None
        self._window_open = False
        title = self.plan.current_step() if self.plan else NO_PLAN
        count = self._failures.get(title, 0) + 1
        self._failures[title] = count
        if count < self._threshold or self.replan_reason:
            return None
        self.replan_reason = f"step {title!r} has failed verification after {count} attempts"
        if self.plan is None:
            ask = "you write a plan with update_plan (you have none yet)"
        else:
            ask = "you call update_plan with a revised plan"
        note = (
            f"That was failed attempt {count} at {title!r} (a change followed by a failing check). Further "
            f"write_file, edit_file and delegate calls will be refused until {ask}. Re-diagnose before editing "
            "again: re-read the failing output, question your assumption about the cause, and change the "
            "approach rather than its details."
        )
        return ReplanTrigger(title, count, note)

    def refusal(self) -> ToolResult | None:
        if not self.replan_reason:
            return None
        return ToolResult.error(
            "replan_required",
            f"Refused: {self.replan_reason}. Call update_plan with a revised plan that changes the approach "
            "before editing again. Reading, searching and running commands are still allowed.",
        )

    def render(self) -> str:
        return self.plan.render() if self.plan else "No plan recorded."
