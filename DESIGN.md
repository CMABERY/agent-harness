# Design

A stdlib-only Python 3.12 harness that takes a natural-language task about a repository and drives a model through planning, reading, editing, running commands and verifying, then returns a structured outcome. About 4,300 lines of source (docstrings and tool descriptions included) and 2,500 lines of tests: 257 tests, about 5 s, fully offline. `mypy --strict` is clean on both. Three independent review passes, each run by an agent that had not seen the code being written, found 31 defects between them. All are fixed and each has a regression test; the decisions below record what those reviews changed.

## Scope

All eleven capabilities are included. Each has tests that exercise it end to end through `ScriptedProvider`.

| # | Capability | Where | Tests |
|---|---|---|---|
| 1 | Provider protocol, `ScriptedProvider`, OpenAI-compatible adapter (urllib) | `providers.py` | `test_providers.py`; `test_openai_wire.py` runs the CLI against a strict local server |
| 2 | Loop with explicit `done` / `blocked` / `budget_exhausted` | `loop.py` | `test_loop.py` |
| 3 | Six workspace tools, schema validation, path containment | `tools.py`, `schema.py`, `workspace.py` | `test_tools.py`, `test_schema.py`, `test_workspace.py` |
| 4 | Untrusted-output delimiting | `untrusted.py` | `test_untrusted.py` |
| 5 | Malformed-output retry, tool errors as results, loop detection | `loop.py` | `test_loop.py` |
| 6 | Verification gate | `ledger.py`, `host.py` | `test_verification.py`, `test_host.py` |
| 7 | JSONL event log, CLI | `events.py`, `cli.py` | `test_events.py`, `test_cli.py`, `test_e2e.py` |
| 8 | Token tracking, compaction | `context.py` | `test_context.py` |
| 9 | Planning with replan enforcement | `plan.py` | `test_plan.py` |
| 10 | Approval policy | `policy.py` | `test_policy.py` |
| 11 | Subagent delegation | `loop.py` (`_delegate`) | `test_delegate.py` |

Not included: streaming responses, parallel tool execution, provider APIs other than OpenAI-compatible chat completions, an OS-level sandbox, Windows support (commands need a POSIX shell and process groups), and git integration beyond what the model does through `run_command`.

## Architecture

Each module exists because it owns a decision. The module docstrings state those decisions; this table is the index.

| Module | Owns |
|---|---|
| `types.py` | The shared vocabulary: messages, tool calls, results, effects, limits, outcomes. Makes no decisions. |
| `schema.py` | Which JSON Schema subset is supported, strict validation, defaults. |
| `workspace.py` | What counts as "inside the workspace", `.git` write protection, atomic writes and file modes, glob semantics. |
| `host.py` | How commands run: `bash -e -o pipefail`, process groups, bounded head+tail capture, non-interactive environment, credential scrubbing. Also the `Clock` seam. |
| `policy.py` | Command classification (read-only, mutating, destructive): a recursive quote- and heredoc-aware scanner in front of `shlex` that fails closed, per-part approval, masked exit status. |
| `providers.py` | The provider contract, wire-format translation, retry and backoff within a deadline. |
| `untrusted.py` | The delimiter protocol for content the harness did not write, and `inline()` for external strings in trusted lines. |
| `ledger.py` | The harness's record of effects. Defines *modification*, *check*, read coverage and the gate. |
| `tools.py` | Tool specs (descriptions written as prompts), argument binding, dispatch, error capture. |
| `plan.py` | Plan state, what an *attempt* is, when a replan is forced. |
| `context.py` | Token estimation in provider units, compaction trigger and size target, tail selection, summary content. |
| `loop.py` | Iteration order, termination, reply acceptance, loop detection, control tools, delegation. |
| `events.py` | The log format and exact replay. |
| `cli.py` | Wiring, flags, exit codes, the operator confirmation prompt. |

### One iteration

```
Agent._run
  1. _check_budgets      model calls (own cap and run-wide cap), tokens (run-wide), wall-clock deadline (run-wide)
                         -> budget_exhausted
  2. _manage_context     estimate (in provider tokens) > 75% of window? compact toward 40% of it, tail sized after
                         fixed overhead; applied only if smaller. Still > window -> budget_exhausted(context)
  3. provider.complete(messages, specs, timeout_s=time left)
                         ProviderError -> wall_clock if the deadline passed, else blocked(provider_error)
  4. record usage        reported, or estimated and flagged; recalibrate the token ratio; ModelCalled event
  5. _unique_ids         repair missing or duplicate call ids (a server artifact the model cannot fix)
  6. _parse              is the whole reply usable? no -> reply discarded, corrective message (merged into a
                         preceding user message), Nth consecutive -> blocked(malformed_output)
  7. append assistant message
  8. for each call, in order:
       loop check        Nth identical consecutive call -> refused with a warning; another identical call in a
                         later reply -> blocked(loop_detected)
       lookup + bind     unknown tool / schema violation -> error result
       finish            must be alone -> ledger.gate() -> done | blocked(unverified)
       replan gate       mutating tool while a replan is required -> error result
       execute           update_plan | delegate (child Agent) | registry.execute(handler)
       _deliver          ledger.record(effects) -> step; plan.observe -> maybe a replan note;
                         render (trusted header + untrusted blocks + notes) -> tool message; ToolExecuted event
```

`Agent.run` wraps `_run`: provider failures, `KeyboardInterrupt` and harness exceptions all become an `Outcome`, and `run_finished` is always the last event.

## Key decisions

**Termination is a tool call, not a text-only reply.** `finish(status, summary, evidence)` yields a structured outcome that is validated by schema. A text-only reply is rejected with "nobody reads plain-text replies; call a tool or finish". *Rejected:* treating a text-only reply as completion, which most chat loops do. That makes the outcome unstructured, ends the run whenever a model pauses to think aloud, and gives the gate nothing to check. *Cost:* one wasted round trip when a model forgets.

**Replies are accepted or rejected whole.** If any call in a reply has unparseable arguments or no name, none of its calls run, the reply is kept out of the conversation, and a corrective message names the problem with an excerpt. It still goes in the event log. *Rejected:* running the parseable calls and erroring the rest. That partially executes a reply from a confused author and leaves the broken text in context to be imitated. Missing or duplicate call ids are *repaired*, not rejected: the server assigns them, so a corrective message could never fix them. Errors *inside* an accepted reply (an unknown tool, a schema violation, a tool failure) go back as that call's result.

**The verification gate runs on harness-observed facts, and exit codes are made trustworthy.** After the last `write_file` or `edit_file`, at least one *check* (a `run_command` that is not read-only) must have passed. Every check that *failed* after that edit must have been re-run with the same command in the same directory and passed. A check whose exit status the policy marks as masked can fail but never pass. Under `-e` the status comes from the last top-level statement, so masked means: the last statement swallows failure (`tests || true`, `! tests`); or the last statement is read-only after an earlier swallowed failure or `set +e` (`tests || true; echo done`); or pipefail was switched off before a final pipeline. `pip install -e . || true; pytest -q` is not masked, because pytest decides its status. Commands run under `bash -e -o pipefail`, the settings GitHub Actions uses, so `tests | tail -50` and `tests; echo "exit=$?"` both report the tests' failure. The model's `evidence` field is recorded as `claimed_evidence` and never trusted; the outcome's `evidence` contains what the harness saw. Read-only commands (`ls`, `cat`, `echo ok`) count neither for nor against. *Rejected, by review:* "any passing command after the last edit". It passed with the prompt's own `| tail -50` idiom, with a failing test followed by `mkdir`, and with `; echo "exit=$?"`. *Rejected:* making the model cite which command verified the change, because string matching against free text is brittle and still gameable. *Rejected:* fingerprinting the workspace around every command to catch edits made through the shell. That falsely blocks any build that writes artifacts, and the model cannot fix such a block. *Accepted gaps:* edits made through `run_command` (`sed -i`) are invisible to the gate (the prompt and tool descriptions route all edits through the file tools); `python -c pass` still counts as a check; and in `pytest || true; ruff check .` only ruff decides the status. The gate is built to catch honest mistakes, not a model set on deceiving it.

**The gate is final.** A failed gate ends the run as `blocked(unverified)`, and the explanation names the file, the step and the failing, masked or missing check. *Rejected:* bouncing `finish` back once as a nudge. That would turn the gate into a suggestion, and the rule is stated in the system prompt, the finish description and the run_command description. `finish` must also be the only call in its reply, so the model has seen every result before it declares an outcome. A combined reply gets `finish_not_alone` and the other calls run.

**Untrusted content gets escaped XML-style delimiters.** File contents, listings, search matches, command output and subagent reports reach the model only inside `<untrusted-data source="...">...</untrusted-data>`. Any occurrence of the tag in content is escaped, case-insensitively, so content cannot close its block or fake a trusted region. The `source` attribute is HTML-escaped because file names are attacker-controlled. Strings that are not the harness's own but must appear on a trusted line (a path in a tool header, which may be a symlink's target name; a subagent's command or file list) go through `inline()`. That escapes line breaks, control characters, undecodable bytes and the tag, so they cannot start a forged `[harness note]` line. *Rejected:* JSON-encoding tool results, which turns code into `\n`-escaped noise that costs tokens and that models read worse. *Rejected:* random-nonce delimiters, because once the tag is escaped a nonce adds nothing. This is structural mitigation, not a guarantee: a model can still choose to obey injected text. The tests therefore assert on what the harness sends.

**Compaction is deterministic, written by the harness, and sized in the provider's units.** When the next prompt would exceed 75% of the window, older turns are replaced by a summary built from the ledger, the plan and an action log, aiming for 40% of the window *including* the system prompt and tool specs (about 4.8k estimated tokens). The verbatim tail gets what remains after that fixed overhead and the summary, starts on a turn boundary, and is at least the last turn. A compaction that would not shrink the prompt is not applied. The original task and the summary share one user message, so the conversation never has two user messages in a row. *Preserved, and why:*

- the system prompt and the task, verbatim, because they are the contract;
- the latest plan with its notes, verbatim, because it is the model's only durable memory;
- each modified file with its last action, line count and hash, so the model knows what it has changed without the content (it is told to re-read before editing, and `edit_file`'s exact-match rule makes stale memory fail loudly instead of corrupting the file);
- the files it has read, so it does not redo the exploration;
- the verification status, because the gate depends on it;
- unresolved failures with their output tail, still delimited as untrusted, because they are the current work;
- a one-line-per-action log of up to 40 earlier actions, to prevent repeated work.

*Rejected:* LLM summarization. A model-written summary of turns that contain injected file text turns that text into trusted prose, which is exactly the laundering the delimiters exist to prevent. It also costs a call, is nondeterministic and cannot be tested offline. *Cost:* the model's free-form reasoning in old turns is lost. The system prompt says so and points to `notes`.

**Token estimates carry the provider's ratio.** The counter is characters / 4 plus per-message overhead, behind `TokenCounter`. When the provider reports `prompt_tokens`, the meter stores the ratio of real to counted tokens (clamped to 0.25–4) and scales every estimate by it: the compaction trigger, the tail budget and the "did it shrink" comparison. *Rejected, by review:* anchoring on the last reported count and adding raw estimates for new messages. A compacted candidate measured in raw units then looked smaller than a calibrated "before", so a compaction that grew the real prompt past the window was applied.

**The replan rule counts attempts, not failures.** An attempt is the window between one modification and the next. It fails if any check inside it fails, so a passing `pip install` does not rescue a failing test. Each window is charged once, to the step in progress, and a subagent's tracker starts at the ledger step when it was created, so the parent's edits are not the child's attempts. Two failed attempts on one step refuse `write_file`, `edit_file` and `delegate` until `update_plan` submits steps with *different titles*; flipping a status does not count. Reading and running commands remain allowed for re-diagnosis. Reproducing a failure before the first edit, and re-running a failing check without editing, are not attempts. *Rejected:* counting every non-zero exit, which punishes reproduction runs and `grep` misses. *Rejected:* enforcing a plan before the first edit, because the harness cannot judge whether a task is trivial. Two failures with no plan is the signal, and the same rule then requires one.

**Delegation shares the ledger and the budgets, and scopes the gate.** A subagent gets a fresh conversation, the same tools minus `delegate` (with a hard stop in `_delegate` as a second guard), and its own iteration cap. Model calls, tokens and the deadline are run-wide, so `--max-iterations 30` means 30 model calls in total, subagents included. Verification is a property of the workspace, so there is one ledger. A child answers for *its own* edits, and the root answers for everything, including the child's edits. A child's check therefore satisfies the parent if nothing changed since, and a child's unverified edit blocks the parent's `done`. The child's summary reaches the parent as untrusted data; the harness-observed facts (files modified, checks) are escaped trusted text. *Rejected:* a separate ledger per child, which would let a parent finish on an edit nobody verified.

**Approval is lexical, fails closed, and is not a sandbox.** A recursive scanner runs before `shlex` and models the shell syntax that changes what runs:

- A `#` is a comment only at the start of a word, outside `${...}`, and only to the end of the line.
- Unquoted newlines separate commands.
- Heredoc delimiters get shell quote removal (`E"OF"` ends at `EOF`). Heredoc bodies are data, unless their line runs a shell (they are classified as commands) or the delimiter is unquoted (their `$(...)` runs and is classified).
- `$(...)`, `${...}`, `$((...))`, backticks and `<(...)` are scanned to their real end, with heredocs and quotes inside them included. That is what makes `git commit -m "$(cat <<'EOF' ... Don't ... EOF)"` parse. Every command substitution is classified as a command.
- Fused operator tokens (`);`, `&&(`, `()`) are split, and `function NAME` and keywords (`then`, `do`, `{`) are stripped.
- Wrappers are unwrapped with their own value options (`env -u`, `nice -n`, `timeout -s`, `xargs -I`, `sh -c`, `bash -lc`, `bash -o pipefail -c`, `eval`, `find -exec`). An unknown program whose trailing words form a destructive command (`setsid git reset --hard`, `docker run … rm -rf /`, `flock l rm -rf $DIR`) takes that classification. A computed word in command position after an unknown program (`setsid $CMD`) is not escalated. Far more often it is data (`pytest $(git diff --name-only)`), so this gap is deliberate.
- Text piped into a shell is classified only when it comes from a literal `echo` (its payload keeps its own flags) or a `printf` without format directives or backslash escapes; anything else piped into a shell is refused.
- Long options match by unambiguous prefix, as git and GNU tools accept them (`git reset --har`).

What the scanner does not model, it refuses: ANSI-C `$'...'` quoting, unbalanced syntax, a program name computed at run time (`$CMD`, `g$(…)`), and computed arguments to programs whose classification depends on them. `rm`, `find`, `dd` and `eval` refuse any computed argument. `git` refuses one unless it is the value of a message option such as `-m`. A shell refuses a computed script, as in `bash <(curl …)`, and so does `curl … | env bash`. Each of these is DESTRUCTIVE, with a reason telling the model to write the value literally. The most severe part decides the kind, and every destructive part is reviewed separately: an allowed `rm -rf build` never approves the `git reset --hard` after it, and `--confirm-destructive` asks the operator once per destructive part, with control characters escaped in the prompt. Allowed-path deletes refuse targets with braces, globs combined with `..`, `~`, variables, or a preceding `cd` / `env -C`. *Rejected:* an allowlist of permitted programs, which is useless for a coding agent whose verification step is "run arbitrary project code". *Rejected:* an OS sandbox, which is out of scope for the standard library. `python -c "import shutil; shutil.rmtree('/')"` is MUTATING and allowed. Run the harness in a container if the model is untrusted.

**Small guards where a mistake is expensive.** Overwriting an existing file is refused (`not_read`) unless the read_file pages seen in this run cover every line, with pages that cut long lines not counting, or the agent itself wrote the whole file. Editing part of a file does not count. A file that is not valid UTF-8 is never overwritten or edited. Writes under any `.git` directory are refused, at any depth and case-insensitively, because a hook written there runs on the user's next commit. Writes are atomic, keep the mode of an existing file, and give new files 0666 minus the umask. `edit_file` counts overlapping occurrences, so `"}\n}"` in `"}\n}\n}"` is ambiguous. It retries a no-match with CRLF line endings for files that use them, and its no-match error points to the line that matches when indentation is ignored. Commands run in their own process group, which is killed when they exit or time out. Their timeout defaults to `Limits.command_timeout_seconds` and is capped by the remaining wall clock. stdin is `/dev/null`, pagers are disabled, and `AGENT_API_KEY`, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are removed from the child environment, so an injected `env | curl` finds nothing to steal. The event log defaults to a path outside the workspace, so the file tools cannot reach it. `run_command` is not sandboxed and could.

**Schema validation is strict.** There is no coercion of `"10"` to `10`, unknown properties are rejected, and every problem is reported in one message. A keyword outside the supported subset, or a default that violates its own schema, fails at registration. *Rejected:* lenient coercion, because a tool that silently ran with misread arguments costs more than one corrective turn.

**The provider adapter is lenient about wire format and strict about time.** It accepts arguments sent as an object, missing or duplicate call ids and content sent as a list of parts, because those are server quirks rather than model errors. It retries 408, 425, 429, 500, 502, 503, 504 and 529, network errors and garbled 200 responses, with exponential backoff that honors `Retry-After`. It does not retry other 4xx responses. The loop passes the remaining wall-clock budget; each attempt's socket timeout and each backoff sleep must fit inside it. It never parses tool arguments: the loop owns the malformed-output policy. No conversation the loop sends has two user messages in a row, an unanswered tool call or an orphan tool result. Strict chat templates (Mistral, Gemma on vLLM) reject those, and `test_openai_wire.py` enforces the same rules with a local server.

**The log replays exactly.** Every message that enters a conversation is a `message_added` event with its index; a merged user message re-uses its index, which is how a replacement is recorded. Compaction logs the full replacement conversation. Lines are ASCII-escaped JSON, so undecodable bytes in a file name cannot make a write fail. `reconstruct_conversation` replays the log and is tested to equal the agent's final conversation, including across compaction and merges. *Rejected:* logging the full request on every call, which grows quadratically.

## Failure modes

| What breaks | What the harness does |
|---|---|
| Invalid JSON arguments, a text-only or empty reply, output cut off at the length limit | The reply is discarded and a corrective message explains why (the cut-off case adds "make smaller edits"). After 3 in a row: `blocked(malformed_output)`. |
| Missing or duplicate call ids | Repaired to unique ids; the reply is accepted. |
| Unknown tool, schema violation | An error result lists the available tools or every schema problem. |
| A tool raises | `internal_error` result with exception and location; the run continues. |
| Provider 429/5xx/network/garbled body | Retried with backoff inside the remaining wall clock. When retries run out: `blocked(provider_error)`, or `budget_exhausted(wall_clock)` if time ran out. A 401 or 400 fails immediately. |
| The same call with the same arguments, repeatedly | The 3rd consecutive identical call is refused with a warning, and so are further copies in the same reply. Repeating it in a later reply ends the run as `blocked(loop_detected)`. Alternating loops (A, B, A, B) are not detected; the run-wide iteration cap is the backstop. |
| A runaway, hung or interactive command | Timeout of min(requested, configured maximum, remaining wall clock); the process group is killed; partial output is returned with `timed out`. stdin is empty, so prompts fail fast. |
| A command line that hides a failure | `pipefail` and `-e` keep `tests \| tail` and `tests; echo` honest; `tests \|\| true` and `tests \|\| true; echo done` are flagged as masked and cannot pass the gate. Under `/bin/sh` without bash there is no pipefail. |
| Huge output | Head and tail capture with bounded memory. The omitted byte count is marked inline, and the trusted header gives the total size and says the stream was truncated. |
| Hostile file contents or file names | Contents are delimited, with the tag escaped and the source escaped. Names on trusted lines go through `inline()`. The system prompt states the contract. Not a guarantee against a model that decides to comply. |
| `..`, absolute paths, symlinks pointing out | `path_outside_workspace`. Symlinks swapped between check and use (TOCTOU) are not defended. |
| Writes to a `.git` directory, overwrite without a full read | `protected_path`, `not_read`. |
| Binary, non-UTF-8, unreadable or very large files | `binary_file`; read with U+FFFD, but `edit_file` and `write_file` refuse (`not_utf8`); `search` skips unreadable files and dangling links and reports how many; `file_too_large` above 10 MB; lines over 2,000 characters are cut, and the cut is reported. |
| Context approaches the window | Compaction, in provider tokens. If even the last turn does not fit: `budget_exhausted(context)` instead of an HTTP 400. Windows below about 9k estimated tokens (under twice the fixed overhead) compact on nearly every call. That works, but it is wasteful. |
| `done` without passing verification after the last edit | `blocked(unverified)` with the explanation and the agent's own summary. |
| Destructive or unreviewable command, alone or in a chain | `refused_by_policy`, telling the model not to work around it and to write values literally. With `--confirm-destructive`, a human decides each destructive part. |
| Two failed fix attempts on one step | Mutating tools are refused until the step titles change. |
| Harness bug in the loop | `blocked(internal_error)`; the traceback is in the log. |
| Ctrl-C | `blocked(interrupted)` from the root; a running command's process group is killed. |
| The event log cannot be written | The run continues, records stay in memory, and the CLI prints a warning. |
| A regex with catastrophic backtracking in `search` | **Not defended**: Python's `re` has no timeout. Files are capped at 2 MB, which bounds the damage. |
| Secrets in workspace files (`.env`) | **Not defended**: the model can read them, and their contents go to the provider and into the log. |

## Running it

```sh
cd agent
python3.12 -m pytest                      # 257 tests, offline, about 5 s; needs only pytest
python3.12 -m mypy agent tests            # optional; strict mode, clean

# Offline demo: a scripted model fixes a failing test in a copy of the demo repo.
cp -r examples/demo_repo /tmp/demo
python3.12 -m agent "Fix the failing tests." --workspace /tmp/demo \
    --provider scripted --script examples/fix_mean.json --log /tmp/demo.jsonl

# A real model, through any OpenAI-compatible chat-completions endpoint.
export AGENT_BASE_URL=https://api.openai.com/v1   # default; e.g. http://localhost:8000/v1 for vLLM
export AGENT_API_KEY=sk-...                       # may be empty for localhost endpoints
export AGENT_MODEL=gpt-4.1
python3.12 -m agent "Make the date parser accept ISO week dates" --workspace ~/src/project \
    --max-iterations 30 --allow-delete build --log run.jsonl
```

Flags: `--provider {openai,scripted}`, `--script`, `--max-iterations` (run-wide), `--max-tokens`, `--max-seconds`, `--context-window`, `--log`, `--allow-delete PATH` (repeatable), `--confirm-destructive`, `--json`. Exit codes: 0 done, 1 blocked, 2 budget exhausted, 64 usage or configuration error. Embedding: build `Agent(provider, Workspace(root), Runtime(clock, runner, policy, events, counter))` and call `.run(task)`. Every collaborator is injected.
