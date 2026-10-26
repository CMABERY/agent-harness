# Self-review

**Fix first, given another hour.** (1) Make edits made through the shell visible to the gate: snapshot size and mtime of `git ls-files` around each mutating command and record any changes as modifications. (2) Detect cycles in tool calls (A, B, A, B), not only consecutive repeats. (3) Run `search` in a subprocess with a timeout, so a regex with catastrophic backtracking cannot hang the harness.

**Absent.** No numbered capability is missing. Outside the scope: streaming, parallel tool execution, an OS sandbox, Windows support, and provider APIs other than OpenAI-compatible chat completions.

**The weakness a reviewer is most likely to find.** The approval policy is a lexical classifier of bash. Most of the 31 defects that three independent reviews found were in it. All are fixed and pinned by tests, but the class is open-ended. The policy fails closed on syntax it does not model, so it guards against mistakes and injected instructions; it is not a sandbox, and `python -c "shutil.rmtree(...)"` passes by design. Close behind: the gate proves that a check passed after the last edit, not that the check exercises the change, so `python -c pass` counts.
