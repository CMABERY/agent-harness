from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agent.context import CharTokenCounter, ContextMeter
from agent.events import reconstruct_conversation
from agent.providers import ProviderRequest, ScriptStep, call, respond
from agent.types import Limits, Message, ModelResponse, TokenUsage, ToolCall, ToolSpec
from tests.fakes import Canned, make_rig, payload


class MessagesOnly(CharTokenCounter):
    """Count messages as usual but treat tool specs as free, so thresholds in
    these tests depend only on the conversation."""

    def count_tools(self, tools: Sequence[ToolSpec]) -> int:
        return 0


def test_meter_scales_every_estimate_by_the_reported_ratio() -> None:
    counter = CharTokenCounter()
    meter = ContextMeter(counter)
    history = [Message("user", "x" * 400)]
    assert meter.estimate(history, ()) == 104
    meter.observe(raw_prompt_tokens=104, reported_prompt_tokens=208)  # the real tokenizer counts twice as many
    assert meter.estimate(history, ()) == 208
    assert meter.estimate([Message("user", "y" * 40)], ()) == 28  # smaller conversations scale the same way
    meter.observe(raw_prompt_tokens=100, reported_prompt_tokens=10_000)
    assert meter.ratio == 4.0  # clamped: one odd report must not make every estimate absurd


def test_compaction_preserves_what_the_agent_needs(repo: Path) -> None:
    (repo / "big.txt").write_text("".join(f"{i:03d} " + "z" * 56 + "\n" for i in range(100)), encoding="utf-8")
    plan = call("update_plan", {
        "acceptance_criteria": ["`pytest -q` passes"],
        "steps": [{"title": "Fix add()", "status": "in_progress"}, {"title": "Run tests", "status": "pending"}],
        "notes": "Root cause: add() subtracts.",
    })
    script = [
        respond(plan),
        respond(call("edit_file", {"path": "src/calc.py", "old_string": "a - b", "new_string": "a + b"})),
        respond(call("run_command", {"command": "pytest -q"})),
        respond(call("read_file", {"path": "big.txt"})),
        respond(call("read_file", {"path": "big.txt", "offset": 2})),
        respond(call("read_file", {"path": "tests/test_calc.py"})),
        respond(call("finish", {"status": "blocked", "summary": "x"})),
    ]
    limits = Limits(context_window_tokens=4000, compact_at_fraction=0.75, compact_to_fraction=0.4)
    rig = make_rig(repo, script, limits=limits, counter=MessagesOnly(),
                   results={"pytest -q": Canned(exit_code=1, stderr="E   assert add(2, 3) == 5")})
    rig.agent.run("Fix add().")

    compactions = [r for r in rig.events.records if r.kind == "context_compacted"]
    assert len(compactions) == 1
    data = compactions[0].data
    assert isinstance(data["tokens_before"], int) and isinstance(data["tokens_after"], int)
    assert data["tokens_before"] > 3000 > data["tokens_after"] and data["removed_messages"] == 8

    sent = rig.sent(5)  # the first call after compaction
    assert [m.role for m in sent] == ["system", "user", "assistant", "tool"]  # no two user messages in a row
    assert sent[0] == rig.sent(0)[0]  # system prompt verbatim
    task = rig.sent(0)[1].content
    assert sent[1].content.startswith(task + "\n\n[harness] Context compacted")  # task verbatim, then summary
    assert sent[2].tool_calls[0].id == sent[3].tool_call_id  # the tail starts on a turn boundary
    summary = sent[1].content
    for expected in (
        "Context compacted: 8 earlier message(s)",
        "Plan v1",
        "[>] Fix add()",
        "Notes: Root cause: add() subtracts.",
        "src/calc.py: edited at step 2",
        "NOT VERIFIED",
        "`pytest -q` exited 1",
        '<untrusted-data source="command output, step 3">\nE   assert add(2, 3) == 5\n</untrusted-data>',
        'update_plan {"acceptance_criteria"',
        '3. run_command {"command": "pytest -q"} -> exit 1',
        "Files read but not modified ==\nbig.txt",
    ):
        assert expected in summary, expected
    assert "z" * 56 not in summary  # file contents are not carried into the summary

    assert reconstruct_conversation(rig.events.records) == rig.agent.messages


def test_context_that_cannot_fit_is_a_budget_outcome(repo: Path) -> None:
    (repo / "big.txt").write_text(("z" * 80 + "\n") * 100, encoding="utf-8")
    script = [respond(call("read_file", {"path": "big.txt"})), respond(call("finish", {"status": "blocked",
                                                                                         "summary": "x"}))]
    rig = make_rig(repo, script, limits=Limits(context_window_tokens=1000), counter=MessagesOnly())
    outcome = rig.agent.run("Read it.")
    assert (outcome.status, outcome.cause) == ("budget_exhausted", "context")
    assert "1000-token context window" in outcome.message


def test_compaction_accounts_for_fixed_overhead_and_never_grows_the_prompt(repo: Path) -> None:
    """Real system prompt and tool specs (about 4.8k estimated tokens) in a 16k window."""
    (repo / "data.txt").write_text("".join(f"row {i:04d} ..........\n" for i in range(300)), encoding="utf-8")
    reads = [respond(call("read_file", {"path": "data.txt", "offset": k, "limit": 280})) for k in range(1, 13)]
    script = [*reads, respond(call("finish", {"status": "blocked", "summary": "x"}))]
    rig = make_rig(repo, script, system_prompt=None, limits=Limits(context_window_tokens=16_000))
    outcome = rig.agent.run("Read the data.")
    assert outcome.cause == "reported"
    compactions = [payload(r) for r in rig.events.records if r.kind == "context_compacted"]
    assert len(compactions) >= 2
    assert all(c["tokens_after"] < c["tokens_before"] for c in compactions)
    assert len(compactions) <= len(reads) // 2  # it frees real room instead of compacting on every call
    assert all(c["tokens_after"] <= 16_000 * 0.75 for c in compactions)
    assert reconstruct_conversation(rig.events.records) == rig.agent.messages


def test_compaction_stays_inside_the_window_when_the_real_tokenizer_counts_more(repo: Path) -> None:
    """The provider reports twice the character estimate; compaction must reason in its units."""
    (repo / "data.txt").write_text("".join(f"row {i:04d} ..........\n" for i in range(300)), encoding="utf-8")
    counter = CharTokenCounter()
    reported: list[int] = []

    def doubled(tool_call: ToolCall) -> ScriptStep:
        def step(request: ProviderRequest) -> ModelResponse:
            prompt = 2 * (counter.count_messages(request.messages) + counter.count_tools(request.tools))
            reported.append(prompt)
            return respond(tool_call, usage=TokenUsage(prompt, 20))
        return step

    reads = [doubled(call("read_file", {"path": "data.txt", "offset": k, "limit": 280})) for k in range(1, 13)]
    finish = doubled(call("finish", {"status": "blocked", "summary": "x"}))
    rig = make_rig(repo, [*reads, finish], system_prompt=None, limits=Limits(context_window_tokens=32_000))
    assert rig.agent.run("Read the data.").cause == "reported"
    compactions = [payload(r) for r in rig.events.records if r.kind == "context_compacted"]
    assert compactions and all(c["tokens_after"] < c["tokens_before"] for c in compactions)
    assert max(reported) <= 32_000  # what the provider actually received never exceeded the window
