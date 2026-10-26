# agent-harness

An autonomous coding-agent harness in standard-library Python 3.12. Give it a natural-language task about a repository and it drives a model through planning, reading, editing, running commands and verifying, then returns a structured outcome: `done`, `blocked` or `budget_exhausted`.

No runtime dependencies. 257 offline tests run in about 5 seconds, and `mypy --strict` is clean.

## What makes it different

- **A verification gate.** The run cannot end `done` unless a real check passed after the last edit. Exit codes are made trustworthy: commands run under `bash -e -o pipefail`, and masked statuses like `tests || true` can fail but never pass.
- **Untrusted content is delimited.** File contents, command output and subagent reports reach the model only inside escaped `<untrusted-data>` blocks, so injected text cannot fake a trusted instruction.
- **Deterministic compaction.** When context fills, the harness writes the summary itself from its own ledger and plan. No model call, and nothing injected gets laundered into trusted prose.
- **An approval policy that fails closed.** A quote- and heredoc-aware shell scanner classifies each command as read-only, mutating or destructive, and refuses syntax it cannot model.
- **Replan enforcement.** Two failed fix attempts on one plan step lock the editing tools until the plan actually changes.
- **Subagent delegation** with shared budgets and one shared verification ledger.
- **An exact-replay JSONL event log.**

[DESIGN.md](DESIGN.md) covers the architecture, each key decision with the alternatives it rejected, and a full failure-mode table.

## Quick start

Requires Python 3.12+ and a POSIX shell (Linux or macOS).

```sh
python3.12 -m pip install pytest
python3.12 -m pytest                      # 257 tests, offline

# Offline demo: a scripted model fixes a failing test in a copy of the demo repo.
cp -r examples/demo_repo /tmp/demo
python3.12 -m agent "Fix the failing tests." --workspace /tmp/demo \
    --provider scripted --script examples/fix_mean.json --log /tmp/demo.jsonl
```

With a real model, through any OpenAI-compatible chat-completions endpoint:

```sh
export AGENT_BASE_URL=https://api.openai.com/v1   # or a local vLLM / llama.cpp server
export AGENT_API_KEY=...
export AGENT_MODEL=gpt-4.1
python3.12 -m agent "Make the date parser accept ISO week dates" --workspace ~/src/project \
    --max-iterations 30 --log run.jsonl
```

Exit codes: `0` done, `1` blocked, `2` budget exhausted, `64` usage error.

## Limits

It is not a sandbox. The approval policy guards against mistakes and injected instructions, not a model determined to cause harm, so run it in a container if the model is untrusted. Streaming, parallel tool calls and Windows are out of scope.

## License

[MIT](LICENSE)
