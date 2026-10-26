"""Command-line entrypoint: ``python -m agent "<task>" --workspace DIR``.

Exit codes: 0 done, 1 blocked, 2 budget exhausted, 64 usage/configuration error.
The event log defaults to a path *outside* the workspace, so the file tools
cannot read or rewrite the record of the run. run_command is not sandboxed, so
a command could still reach it; keep logs somewhere the agent's user cannot
write if that matters.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from agent.context import CharTokenCounter
from agent.events import EventLog
from agent.host import SubprocessRunner, SystemClock
from agent.loop import Agent, Runtime
from agent.policy import Classification, Confirm, DefaultPolicy
from agent.providers import LLMProvider, OpenAIChatProvider, OpenAIConfig, ProviderError, ScriptedProvider
from agent.types import Limits, Outcome
from agent.untrusted import printable
from agent.workspace import Workspace

EXIT_CODES = {"done": 0, "blocked": 1, "budget_exhausted": 2}
USAGE_ERROR = 64


def build_parser() -> argparse.ArgumentParser:
    defaults = Limits()
    parser = argparse.ArgumentParser(prog="agent", description="Run an autonomous coding agent on a repository.")
    parser.add_argument("task", help="What the agent should do, in natural language.")
    parser.add_argument("--workspace", "-w", type=Path, required=True,
                        help="Repository root. File tools cannot reach outside it.")
    parser.add_argument("--provider", choices=["openai", "scripted"], default="openai",
                        help="openai: an OpenAI-compatible chat-completions endpoint configured by AGENT_BASE_URL, "
                             "AGENT_API_KEY and AGENT_MODEL. scripted: replay --script offline.")
    parser.add_argument("--script", type=Path, help="JSON script of model responses for --provider scripted.")
    parser.add_argument("--max-iterations", type=int,
                        help=f"Model-call cap for the whole run, subagents included "
                             f"(default {defaults.max_iterations}).")
    parser.add_argument("--max-tokens", type=int, help=f"Total token cap (default {defaults.max_total_tokens}).")
    parser.add_argument("--max-seconds", type=float, help=f"Wall-clock cap (default {defaults.max_wall_seconds:.0f}).")
    parser.add_argument("--context-window", type=int,
                        help=f"Context window in tokens; compaction starts at "
                             f"{round(defaults.compact_at_fraction * 100)}%% of it "
                             f"(default {defaults.context_window_tokens}).")
    parser.add_argument("--log", type=Path, help="JSONL event log path (default: under ~/.local/state/agent/runs).")
    parser.add_argument("--allow-delete", action="append", default=[], metavar="PATH",
                        help="Workspace-relative path where recursive deletes are allowed. Repeatable.")
    parser.add_argument("--confirm-destructive", action="store_true",
                        help="Ask on the terminal before a destructive command instead of refusing it.")
    parser.add_argument("--json", action="store_true", help="Print the outcome as JSON.")
    return parser


def default_log_path(environ: Mapping[str, str]) -> Path:
    state = environ.get("XDG_STATE_HOME") or str(Path(environ.get("HOME", ".")) / ".local" / "state")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(state) / "agent" / "runs" / f"run-{stamp}-{os.getpid()}.jsonl"


def terminal_confirm(stdin: TextIO, stderr: TextIO) -> Confirm:
    """Ask once per destructive part. The command is shown escaped, so a carriage
    return or terminal escape in it cannot hide what is really being approved."""

    def confirm(command: str, classification: Classification) -> bool:
        stderr.write(f"\nThe agent wants to run a command containing a destructive part:\n"
                     f"    {printable(command)}\nDestructive part: {printable(classification.reason)}\n"
                     "Allow this part? [y/N] ")
        stderr.flush()
        return stdin.readline().strip().lower() in {"y", "yes"}

    return confirm


def _provider(args: argparse.Namespace, environ: Mapping[str, str]) -> LLMProvider:
    if args.provider == "scripted":
        if args.script is None:
            raise ValueError("--provider scripted requires --script FILE")
        return ScriptedProvider.from_json(json.loads(Path(args.script).read_text(encoding="utf-8")))
    return OpenAIChatProvider(OpenAIConfig.from_env(environ))


def _limits(args: argparse.Namespace) -> Limits:
    overrides: dict[str, float | int] = {}
    for flag, field in (("max_iterations", "max_iterations"), ("max_tokens", "max_total_tokens"),
                        ("max_seconds", "max_wall_seconds"), ("context_window", "context_window_tokens")):
        value = getattr(args, flag)
        if value is not None:
            if value <= 0:
                raise ValueError(f"--{flag.replace('_', '-')} must be positive")
            overrides[field] = value
    return dataclasses.replace(Limits(), **overrides)  # type: ignore[arg-type]


def render_outcome(outcome: Outcome, log_path: Path) -> str:
    lines = [f"status:   {outcome.status} ({outcome.cause})", f"message:  {outcome.message}"]
    for item in outcome.evidence:
        lines.append(f"evidence: step {item.step}: `{item.command}` (exit {item.exit_code})")
    if outcome.files_modified:
        lines.append(f"files:    {', '.join(outcome.files_modified)}")
    lines.append(f"usage:    {outcome.iterations} model call(s), {outcome.usage.total} tokens "
                 f"({outcome.usage.prompt_tokens} prompt + {outcome.usage.completion_tokens} completion)")
    lines.append(f"log:      {log_path}")
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    env = os.environ if environ is None else environ
    out, err = stdout or sys.stdout, stderr or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        workspace = Workspace(args.workspace)
        provider = _provider(args, env)
        limits = _limits(args)
    except (OSError, ValueError, ProviderError) as exc:
        print(f"agent: {exc}", file=err)
        return USAGE_ERROR
    clock = SystemClock()
    log_path: Path = args.log or default_log_path(env)
    events = EventLog(log_path, clock)
    confirm = terminal_confirm(stdin or sys.stdin, err) if args.confirm_destructive else None
    runtime = Runtime(
        clock=clock,
        runner=SubprocessRunner(env),
        policy=DefaultPolicy(workspace.root, args.allow_delete, confirm),
        events=events,
        counter=CharTokenCounter(),
    )
    outcome = Agent(provider, workspace, runtime, limits=limits).run(args.task)
    events.close()
    if args.json:
        print(json.dumps({**dataclasses.asdict(outcome), "log": str(log_path)}, indent=2), file=out)
    else:
        print(render_outcome(outcome, log_path), file=out)
    if events.write_error:
        print(f"agent: warning: {events.write_error}", file=err)
    return EXIT_CODES[outcome.status]
