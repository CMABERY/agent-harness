"""The agent loop: model -> tool calls -> results -> model, until an outcome.

Decisions owned here:
  * Termination is a value, not an exception. ``Agent.run`` always returns an
    ``Outcome``; provider failures, harness bugs and Ctrl-C included.
  * Budgets (model calls, total tokens, wall clock) are checked before every
    model call. They are run-wide: a subagent's calls and tokens count against
    the caller's caps, and it also has its own smaller iteration cap. Model
    calls and commands get at most the remaining wall-clock time.
  * A reply is accepted or rejected as a whole. If any tool call in it has no
    name or arguments that are not a JSON object, none of its calls run, it is
    not added to the conversation, and a corrective message explains the
    problem. Missing or duplicate call ids are repaired rather than rejected:
    the server assigns them, so the model could not fix them. A text-only reply is rejected the same way: nobody
    reads prose mid-run. ``max_consecutive_malformed`` rejections in a row end
    the run as blocked. A corrective message that would follow another user
    message is merged into it, because some chat templates reject two user
    messages in a row.
  * Errors *inside* an accepted reply (unknown tool, schema violation, a tool
    failure) are returned to the model as that call's result.
  * Loop detection: the Nth identical consecutive call (same tool, same
    canonical arguments) is not run and returns a warning; one more identical
    call ends the run as blocked.
  * ``finish`` must be alone in its reply, so the model has seen every result
    before it declares an outcome. ``done`` passes through the verification
    gate (ledger.py) or becomes ``blocked``.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, replace
from importlib import resources

from agent.context import ActionRecord, ContextMeter, TokenCounter, abbreviate, build_summary, compact, select_tail
from agent.events import (
    ContextCompacted,
    Event,
    InternalError,
    LoopDetected,
    MalformedOutput,
    MessageAdded,
    ModelCalled,
    PlanUpdated,
    ReplanRequired,
    RunFinished,
    RunStarted,
    ToolExecuted,
    VerificationDowngraded,
)
from agent.events import EventLog
from agent.host import Clock, CommandRunner
from agent.ledger import Ledger
from agent.plan import PlanTracker
from agent.policy import ApprovalPolicy
from agent.providers import LLMProvider, ProviderError
from agent.schema import ToolArgs
from agent.tools import DELEGATE, FINISH, UPDATE_PLAN, ToolContext, build_registry
from agent.types import (
    JSONObject,
    Limits,
    Message,
    ModelResponse,
    Outcome,
    OutcomeCause,
    OutcomeStatus,
    TokenUsage,
    ToolCall,
    ToolResult,
    UntrustedText,
    VerificationEvidence,
)
from agent.untrusted import inline, printable, render_tool_result
from agent.workspace import Workspace


def load_prompt(name: str) -> str:
    return resources.files("agent").joinpath("prompts", name).read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Runtime:
    """Injected dependencies, shared by an agent and its subagents."""

    clock: Clock
    runner: CommandRunner
    policy: ApprovalPolicy
    events: EventLog
    counter: TokenCounter


@dataclass(slots=True)
class SharedRun:
    """State owned by the whole run rather than by one agent."""

    ledger: Ledger
    deadline: float  # monotonic
    max_total_tokens: int
    max_model_calls: int
    usage: TokenUsage
    model_calls: int = 0


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        workspace: Workspace,
        runtime: Runtime,
        *,
        limits: Limits | None = None,
        system_prompt: str | None = None,
        name: str = "main",
        depth: int = 0,
        shared: SharedRun | None = None,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.runtime = runtime
        self.limits = limits or Limits()
        self.name = name
        self.depth = depth
        self.system_prompt = system_prompt if system_prompt is not None else load_prompt("system.md")
        self.shared = shared or SharedRun(
            ledger=Ledger(),
            deadline=runtime.clock.monotonic() + self.limits.max_wall_seconds,
            max_total_tokens=self.limits.max_total_tokens,
            max_model_calls=self.limits.max_iterations,
            usage=TokenUsage(),
        )
        self.registry = build_registry(self.limits, allow_delegate=depth == 0)
        self.plan = PlanTracker(self.limits.replan_after_failed_attempts, start_step=self.shared.ledger.step)
        self.meter = ContextMeter(runtime.counter)
        self.messages: list[Message] = []
        self.actions: list[ActionRecord] = []
        self.iterations = 0
        self.usage = TokenUsage()
        self._malformed = 0
        self._last_signature: tuple[str, str] | None = None
        self._repeats = 0
        self._warned_iteration: int | None = None  # the reply in which the current streak was warned about
        self._delegations = 0
        self._generated_ids = 0
        self._task_text = ""
        # A subagent answers for its own edits; the root answers for the workspace.
        self._gate_scope = None if depth == 0 else name

    # ------------------------------------------------------------------ run

    def run(self, task: str) -> Outcome:
        try:
            outcome = self._run(task)
        except KeyboardInterrupt:
            if self.depth > 0:
                raise  # the operator wants the whole run to stop, not just this subagent
            outcome = self._outcome("blocked", "interrupted", "Interrupted by the operator.")
        except Exception as exc:  # a harness bug; still return a structured outcome
            self._emit(InternalError("loop", f"{type(exc).__name__}: {exc}", traceback.format_exc()))
            outcome = self._outcome("blocked", "internal_error", f"Harness error: {type(exc).__name__}: {exc}")
        self._emit(RunFinished(outcome))
        return outcome

    def _run(self, task: str) -> Outcome:
        self._emit(RunStarted(task, str(self.workspace.root), self.depth, self.registry.names(), self.limits))
        self._task_text = self._task_message(task)
        self._append(Message("system", self.system_prompt))
        self._append(Message("user", self._task_text))
        while True:
            stop = self._check_budgets() or self._manage_context()
            if stop is not None:
                return stop
            response = self._call_model()
            if isinstance(response, Outcome):
                return response
            response = self._unique_ids(response)
            parsed = self._parse(response)
            if isinstance(parsed, str):
                stop = self._reject(response, parsed)
                if stop is not None:
                    return stop
                continue
            self._malformed = 0
            self._append(Message("assistant", response.text, response.tool_calls))
            for call, arguments in parsed:
                stop = self._execute(call, arguments, alone=len(parsed) == 1)
                if stop is not None:
                    return stop

    def _task_message(self, task: str) -> str:
        budget = min(self.limits.max_iterations, self.shared.max_model_calls - self.shared.model_calls)
        return (
            f"# Task\n{task.strip()}\n\n# Environment\n"
            f"Workspace root: {self.workspace.root}. Tool paths are relative to it, and commands run there "
            f"unless you pass cwd.\nBudget: at most {budget} model calls for this task."
        )

    # -------------------------------------------------------------- budgets

    def _check_budgets(self) -> Outcome | None:
        progress = self.shared.ledger.verification_status(self._gate_scope)
        if self.iterations >= self.limits.max_iterations:
            return self._outcome("budget_exhausted", "iterations",
                                 f"Iteration cap reached ({self.limits.max_iterations} model calls). {progress}")
        if self.shared.model_calls >= self.shared.max_model_calls:
            return self._outcome("budget_exhausted", "iterations",
                                 f"Run-wide iteration cap reached ({self.shared.max_model_calls} model calls, "
                                 f"subagents included). {progress}")
        if self.shared.usage.total >= self.shared.max_total_tokens:
            return self._outcome("budget_exhausted", "tokens",
                                 f"Token cap reached ({self.shared.usage.total} of "
                                 f"{self.shared.max_total_tokens}). {progress}")
        if self.runtime.clock.monotonic() >= self.shared.deadline:
            return self._outcome("budget_exhausted", "wall_clock",
                                 f"Wall-clock cap reached ({self.limits.max_wall_seconds:.0f}s). {progress}")
        return None

    def _manage_context(self) -> Outcome | None:
        window = self.limits.context_window_tokens
        specs = self.registry.specs()
        counter = self.runtime.counter
        estimate = self.meter.estimate(self.messages, specs)
        if estimate <= int(window * self.limits.compact_at_fraction):
            return None
        plan, ledger = self.plan.render(), self.shared.ledger
        # Size the verbatim tail in counter units: the target, converted back from
        # provider tokens, minus the fixed overhead and the summary.
        overhead = counter.count_messages([self.messages[0], Message("user", self._task_text)]) + counter.count_tools(specs)
        summary_size = counter.count_text(build_summary(len(self.messages), plan, ledger, self.actions, self._gate_scope))
        target = int(window * self.limits.compact_to_fraction / self.meter.ratio)
        start = select_tail(self.messages, max(target - overhead - summary_size, 0), counter)
        if start is not None:
            removed = start - 2
            summary = build_summary(removed, plan, ledger, self.actions, self._gate_scope)
            candidate = compact(self.messages, start, self._task_text, summary)
            after = self.meter.estimate(candidate, specs)  # same scale as `estimate`
            if after < estimate:  # never "compact" into a larger prompt
                self.messages = candidate
                self._emit(ContextCompacted(estimate, after, removed, tuple(self.messages)))
                estimate = after
        if estimate > window:
            return self._outcome(
                "budget_exhausted", "context",
                f"The conversation needs about {estimate} tokens even after compaction, more than the "
                f"{window}-token context window. {self.shared.ledger.verification_status(self._gate_scope)}",
            )
        return None

    # ---------------------------------------------------------- model calls

    def _call_model(self) -> ModelResponse | Outcome:
        specs = self.registry.specs()
        raw_prompt = self.meter.raw(self.messages, specs)
        prompt_estimate = self.meter.estimate(self.messages, specs)
        started = self.runtime.clock.monotonic()
        try:
            response = self.provider.complete(list(self.messages), specs,
                                              timeout_s=max(self.shared.deadline - started, 0.0))
        except ProviderError as exc:
            if self.runtime.clock.monotonic() >= self.shared.deadline:
                return self._outcome("budget_exhausted", "wall_clock",
                                     f"Wall-clock cap reached during a model call ({exc}). "
                                     f"{self.shared.ledger.verification_status(self._gate_scope)}")
            return self._outcome("blocked", "provider_error", f"The model provider failed: {exc}")
        self.iterations += 1
        self.shared.model_calls += 1
        reported = response.usage
        usage = reported or TokenUsage(
            prompt_estimate,
            self.runtime.counter.count_text(response.text + "".join(c.name + c.arguments for c in response.tool_calls)),
        )
        self.meter.observe(raw_prompt, reported.prompt_tokens if reported else None)
        self.usage += usage
        self.shared.usage += usage
        self._emit(ModelCalled(self.iterations, self.runtime.clock.monotonic() - started, usage, reported is None,
                               prompt_estimate, response.finish_reason, len(response.tool_calls)))
        return response

    def _unique_ids(self, response: ModelResponse) -> ModelResponse:
        seen: set[str] = set()
        calls: list[ToolCall] = []
        for call in response.tool_calls:
            if not call.id or call.id in seen:
                self._generated_ids += 1
                call = replace(call, id=f"{self.name}_call_{self._generated_ids}")
            seen.add(call.id)
            calls.append(call)
        return replace(response, tool_calls=tuple(calls))

    def _parse(self, response: ModelResponse) -> list[tuple[ToolCall, JSONObject]] | str:
        """Return the reply's calls with parsed arguments, or a description of why it is unusable."""
        cut_off = response.finish_reason == "length"
        if not response.tool_calls:
            if cut_off:
                return ("Your reply was cut off at the output-token limit before it reached a tool call. Be "
                        "briefer, and make large changes as several smaller edit_file calls.")
            if not response.text.strip():
                return "Your reply was empty: it contained neither text nor a tool call."
            return ("Your reply contained no tool call. Nobody reads plain-text replies during the run; there "
                    "is no human in the loop. Continue by calling a tool, or end the task by calling finish with "
                    "status \"done\" or \"blocked\" and your final answer in summary.")
        parsed: list[tuple[ToolCall, JSONObject]] = []
        total = len(response.tool_calls)
        for position, call in enumerate(response.tool_calls, 1):
            label = f"tool call {position} of {total} ({printable(call.name) or 'unnamed'})"
            if not call.name:
                return f"The {label} has no tool name."
            try:
                value = json.loads(call.arguments)
            except json.JSONDecodeError as exc:
                hint = (" The reply was also cut off at the output-token limit, which truncates arguments; "
                        "make smaller edits.") if cut_off else ""
                return (f"The arguments of the {label} are not valid JSON ({exc.msg} at character {exc.pos}): "
                        f"{abbreviate(call.arguments, 200)}{hint}")
            if not isinstance(value, dict):
                return f"The arguments of the {label} must be a JSON object, not {type(value).__name__}."
            parsed.append((call, value))
        return parsed

    def _reject(self, response: ModelResponse, problem: str) -> Outcome | None:
        self._malformed += 1
        self._emit(MalformedOutput(self.iterations, self._malformed, problem, response))
        if self._malformed >= self.limits.max_consecutive_malformed:
            return self._outcome("blocked", "malformed_output",
                                 f"The model produced {self._malformed} unusable replies in a row. Last problem: {problem}")
        lead = ("Your last reply was discarded and none of its tool calls ran."
                if response.tool_calls else "Your last reply was not accepted.")
        remaining = self.limits.max_consecutive_malformed - self._malformed
        self._append_user(f"[harness] {lead} {problem} ({remaining} more unusable "
                          f"repl{'y' if remaining == 1 else 'ies'} in a row will end the run.)")
        return None

    # ----------------------------------------------------------- tool calls

    def _execute(self, call: ToolCall, arguments: JSONObject, *, alone: bool) -> Outcome | None:
        signature = (call.name, json.dumps(arguments, sort_keys=True))
        if signature == self._last_signature:
            self._repeats += 1
        else:
            self._repeats, self._warned_iteration = 1, None
        self._last_signature = signature
        threshold = self.limits.loop_repeat_threshold
        if self._repeats >= threshold:
            # End the run only if the model already saw a warning in an earlier reply;
            # identical calls within the reply that triggered the warning are just refused.
            terminated = self._warned_iteration is not None and self._warned_iteration < self.iterations
            if self._warned_iteration is None:
                self._warned_iteration = self.iterations
            self._emit(LoopDetected(call.name, call.arguments, self._repeats, terminated))
            if terminated:
                return self._outcome(
                    "blocked", "loop_detected",
                    f"Stopped: the agent made the identical call {call.name} {abbreviate(call.arguments)} "
                    f"{self._repeats} times in a row, including once after a warning.",
                )
            self._deliver(call, ToolResult.error(
                "loop_detected",
                f"You have made this exact call ({call.name} with identical arguments) {self._repeats} times in a "
                "row. It was not run again: its result would not change. Do something different: re-read the last "
                "result, try another approach, or call finish with status \"blocked\" if you are stuck. Making "
                "this identical call once more ends the run.",
            ), 0.0)
            return None

        tool = self.registry.get(call.name)
        if tool is None:
            self._deliver(call, self.registry.unknown(call.name), 0.0)
            return None
        bound = self.registry.bind(tool, arguments)
        if isinstance(bound, ToolResult):
            self._deliver(call, bound, 0.0)
            return None
        if call.name == FINISH:
            if not alone:
                self._deliver(call, ToolResult.error(
                    "finish_not_alone",
                    "finish must be the only tool call in its reply, so that you have seen the result of every "
                    "action before declaring an outcome. It was not processed; the other calls in this reply are "
                    "processed normally. Check their results, then call finish on its own.",
                ), 0.0)
                return None
            return self._finish(bound)
        refusal = self.plan.refusal() if tool.mutates else None
        if refusal is not None:
            self._deliver(call, refusal, 0.0)
            return None

        started = self.runtime.clock.monotonic()
        if call.name == UPDATE_PLAN:
            result = self.plan.update(bound)
            if result.ok and self.plan.plan is not None:
                self._emit(PlanUpdated(self.plan.plan.version, self.plan.render()))
        elif call.name == DELEGATE:
            result = self._delegate(bound)
        else:
            result = self.registry.execute(tool, bound, self._tool_context())
        self._deliver(call, result, self.runtime.clock.monotonic() - started)
        return None

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            workspace=self.workspace,
            runner=self.runtime.runner,
            policy=self.runtime.policy,
            ledger=self.shared.ledger,
            limits=self.limits,
            clock=self.runtime.clock,
            deadline=self.shared.deadline,
        )

    def _deliver(self, call: ToolCall, result: ToolResult, duration_s: float) -> None:
        step = self.shared.ledger.record(self.name, result.effects)
        notes: list[str] = []
        trigger = self.plan.observe(step, result.effects, self.shared.ledger)
        if trigger is not None:
            notes.append(trigger.note)
            self._emit(ReplanRequired(trigger.step, trigger.failed_attempts))
        command = result.effects.command
        if command is not None:
            label = "timed out" if command.timed_out else f"exit {command.exit_code}"
        else:
            label = "ok" if result.ok else f"error {result.error_code}"
        self.actions.append(ActionRecord(step, call.name, abbreviate(call.arguments), label))
        self._emit(ToolExecuted(call.id, call.name, call.arguments, result.ok, result.error_code,
                                round(duration_s, 6), result.effects))
        self._append(Message("tool", render_tool_result(call.name, result, notes), tool_call_id=call.id))

    # ----------------------------------------------------- control tools

    def _finish(self, args: ToolArgs) -> Outcome:
        summary = args.text("summary").strip()
        claimed = tuple(args.text_list("evidence"))
        if args.text("status") == "blocked":
            return self._outcome("blocked", "reported", summary, claimed=claimed)
        gate = self.shared.ledger.gate(self._gate_scope)
        if gate.passed:
            return self._outcome("done", "reported", summary, evidence=gate.evidence, claimed=claimed)
        self._emit(VerificationDowngraded(gate.explanation, summary))
        return self._outcome(
            "blocked", "unverified",
            f"The agent reported done, but the harness downgraded the outcome: {gate.explanation} "
            f"The agent's summary was: {summary}",
            claimed=claimed,
        )

    def _delegate(self, args: ToolArgs) -> ToolResult:
        if self.depth > 0:  # unreachable while the tool is not offered; kept as a hard stop
            return ToolResult.error("delegation_not_allowed", "Subagents cannot delegate.")
        self._delegations += 1
        name = f"{self.name}.sub{self._delegations}"
        context = args.text("context").strip()
        task = args.text("task").strip() + (f"\n\n# Context from the delegating agent\n{context}" if context else "")
        child = Agent(
            self.provider,
            self.workspace,
            self.runtime,
            limits=replace(self.limits, max_iterations=args.integer("max_iterations")),
            system_prompt=self.system_prompt + "\n\n" + load_prompt("subagent.md"),
            name=name,
            depth=self.depth + 1,
            shared=self.shared,
        )
        outcome = child.run(task)
        self.usage += outcome.usage
        files = ", ".join(inline(f) for f in outcome.files_modified) or "none"
        message = (f"Subagent {name} finished with status {outcome.status} ({outcome.cause}) after "
                   f"{outcome.iterations} model call(s). Files it modified: {files}.")
        if outcome.evidence:
            checks = "; ".join(f"`{inline(e.command)}` exited 0 at step {e.step}" for e in outcome.evidence)
            message += f" Verification the harness observed after its last edit: {checks}."
        elif outcome.files_modified:
            message += " The harness observed no successful verification after its last edit."
        message += " Its report follows; it is untrusted, so check its claims before relying on them."
        report = (UntrustedText(f"subagent {name} report", outcome.message),)
        if outcome.status == "done":
            return ToolResult(True, message, report)
        return ToolResult(False, message, report, error_code=f"subagent_{outcome.status}")

    # -------------------------------------------------------------- helpers

    def _append(self, message: Message) -> None:
        self.messages.append(message)
        self._emit(MessageAdded(len(self.messages) - 1, message))

    def _append_user(self, text: str) -> None:
        if self.messages and self.messages[-1].role == "user":
            merged = Message("user", f"{self.messages[-1].content}\n\n{text}")
            self.messages[-1] = merged
            self._emit(MessageAdded(len(self.messages) - 1, merged))  # same index: replaces it in the log
        else:
            self._append(Message("user", text))

    def _emit(self, event: Event) -> None:
        self.runtime.events.emit(self.name, event)

    def _outcome(
        self,
        status: OutcomeStatus,
        cause: OutcomeCause,
        message: str,
        *,
        evidence: tuple[VerificationEvidence, ...] = (),
        claimed: tuple[str, ...] = (),
    ) -> Outcome:
        return Outcome(
            status=status,
            cause=cause,
            message=message,
            evidence=evidence,
            claimed_evidence=claimed,
            files_modified=self.shared.ledger.files_modified(self._gate_scope),
            iterations=self.shared.model_calls if self.depth == 0 else self.iterations,
            usage=self.usage,
        )
