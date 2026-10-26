from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent.ledger import Ledger
from agent.policy import DefaultPolicy
from agent.schema import ToolArgs
from agent.tools import ToolContext, build_registry
from agent.types import CommandKind, JSONObject, Limits, ReadSpan, ToolResult
from agent.workspace import Workspace
from tests.fakes import Canned, FakeClock, FakeRunner


class Harness:
    def __init__(self, root: Path, results: dict[str, Canned | list[Canned]] | None = None) -> None:
        self.clock = FakeClock()
        self.runner = FakeRunner(self.clock, results or {})
        self.ledger = Ledger()
        self.registry = build_registry(Limits(), allow_delegate=False)
        workspace = Workspace(root)
        self.ctx = ToolContext(workspace, self.runner, DefaultPolicy(workspace.root), self.ledger, Limits(),
                               self.clock, deadline=self.clock.monotonic() + 600)

    def __call__(self, name: str, args: JSONObject) -> ToolResult:
        tool = self.registry.get(name)
        assert tool is not None
        bound = self.registry.bind(tool, args)
        if not isinstance(bound, ToolArgs):
            return bound
        result = self.registry.execute(tool, bound, self.ctx)
        self.ledger.record("main", result.effects)
        return result


@pytest.fixture
def tool(repo: Path) -> Harness:
    return Harness(repo)


def test_invalid_arguments_are_rejected_before_execution(tool: Harness) -> None:
    result = tool("read_file", {"file_path": "src/calc.py"})
    assert not result.ok and result.error_code == "invalid_arguments"
    assert "path: missing required property" in result.message
    assert "file_path: unexpected property" in result.message


def test_read_file_numbers_lines_and_pages(tool: Harness, repo: Path) -> None:
    (repo / "long.txt").write_text("".join(f"line {i}\n" for i in range(1, 11)), encoding="utf-8")
    result = tool("read_file", {"path": "long.txt", "offset": 3, "limit": 2})
    assert result.ok
    assert result.data[0].source == "file:long.txt"
    assert result.data[0].text == "     3\tline 3\n     4\tline 4"
    assert "lines 3-4 of 10" in result.message and "offset=5" in result.message
    assert result.effects.reads == (ReadSpan("long.txt", 3, 4, 10),)


@pytest.mark.parametrize(
    ("setup", "path", "code"),
    [
        (None, "missing.py", "not_found"),
        (None, "src", "is_directory"),
        (b"\x00\x01binary", "blob.bin", "binary_file"),
        (None, "../etc/passwd", "path_outside_workspace"),
    ],
)
def test_read_file_failures(tool: Harness, repo: Path, setup: bytes | None, path: str, code: str) -> None:
    if setup is not None:
        (repo / path).write_bytes(setup)
    result = tool("read_file", {"path": path})
    assert not result.ok and result.error_code == code


def test_write_file_creates_and_refuses_blind_overwrite(tool: Harness, repo: Path) -> None:
    created = tool("write_file", {"path": "pkg/new.py", "content": "x = 1\n"})
    assert created.ok and (repo / "pkg" / "new.py").read_text() == "x = 1\n"
    assert created.effects.changes[0].action == "created"

    blind = tool("write_file", {"path": "src/calc.py", "content": "oops\n"})
    assert blind.error_code == "not_read"
    assert "return a - b" in (repo / "src" / "calc.py").read_text()

    tool("read_file", {"path": "src/calc.py"})
    assert tool("write_file", {"path": "src/calc.py", "content": "ok\n"}).ok


def test_edit_file_exact_replace(tool: Harness, repo: Path) -> None:
    result = tool("edit_file", {"path": "src/calc.py", "old_string": "a - b", "new_string": "a + b"})
    assert result.ok, result.message
    assert (repo / "src" / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    change = result.effects.changes[0]
    assert change.action == "edited" and change.lines == 2 and "line 2" in change.detail


def test_edit_file_fails_loudly_on_absent_match(tool: Harness, repo: Path) -> None:
    before = (repo / "src" / "calc.py").read_text()
    result = tool("edit_file", {"path": "src/calc.py", "old_string": "return a * b", "new_string": "x"})
    assert result.error_code == "no_match" and "unchanged" in result.message
    indented = tool("edit_file", {"path": "src/calc.py", "old_string": "return a - b\n  ", "new_string": "x"})
    assert indented.error_code == "no_match" and "line(s) 2" in indented.message
    prefixed = tool("edit_file", {"path": "src/calc.py", "old_string": "     2\treturn a - b", "new_string": "x"})
    assert "line-number prefixes" in prefixed.message
    assert (repo / "src" / "calc.py").read_text() == before


def test_edit_file_fails_loudly_on_ambiguous_match(tool: Harness, repo: Path) -> None:
    (repo / "dup.py").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
    result = tool("edit_file", {"path": "dup.py", "old_string": "x = 1", "new_string": "x = 9"})
    assert result.error_code == "ambiguous_match"
    assert "2 times" in result.message and "lines 1, 3" in result.message
    assert (repo / "dup.py").read_text() == "x = 1\ny = 2\nx = 1\n"
    every = tool("edit_file", {"path": "dup.py", "old_string": "x = 1", "new_string": "x = 9", "replace_all": True})
    assert every.ok and (repo / "dup.py").read_text() == "x = 9\ny = 2\nx = 9\n"


def test_edit_file_handles_crlf_and_refuses_non_utf8(tool: Harness, repo: Path) -> None:
    (repo / "win.txt").write_bytes(b"one\r\ntwo\r\n")
    assert tool("edit_file", {"path": "win.txt", "old_string": "one\ntwo", "new_string": "1\n2"}).ok
    assert (repo / "win.txt").read_bytes() == b"1\r\n2\r\n"
    (repo / "latin.txt").write_bytes(b"caf\xe9\n")
    assert tool("edit_file", {"path": "latin.txt", "old_string": "caf", "new_string": "tea"}).error_code == "not_utf8"
    assert tool("edit_file", {"path": "win.txt", "old_string": "1", "new_string": "1"}).error_code == "no_change"


def test_list_files_globs_skips_ignored_and_wraps_names(tool: Harness, repo: Path) -> None:
    (repo / "node_modules" / "dep").mkdir(parents=True)
    (repo / "node_modules" / "dep" / "index.py").write_text("", encoding="utf-8")
    result = tool("list_files", {"pattern": "**/*.py"})
    assert result.ok
    assert result.data[0].text.splitlines() == ["src/calc.py", "tests/test_calc.py"]
    limited = tool("list_files", {"pattern": "**/*", "max_results": 1})
    assert "More files match" in limited.message


def test_search_finds_lines_and_reports_bad_regex(tool: Harness) -> None:
    result = tool("search", {"pattern": r"def \w+", "glob": "**/*.py"})
    assert result.ok
    assert result.data[0].text.splitlines() == ["src/calc.py:1: def add(a, b):", "tests/test_calc.py:4: def test_add():"]
    assert tool("search", {"pattern": "("}).error_code == "invalid_regex"
    assert tool("search", {"pattern": "nothing-matches-this"}).data == ()


def test_run_command_reports_exit_code_streams_and_classification(repo: Path) -> None:
    tool = Harness(repo, {"pytest -q": Canned(exit_code=1, stdout="1 failed", stderr="E assert -1 == 5")})
    result = tool("run_command", {"command": "pytest -q"})
    assert result.ok  # the command ran; its failure is data
    assert "Exit code 1 (failure)" in result.message
    assert [d.source for d in result.data] == ["command stdout", "command stderr"]
    run = result.effects.command
    assert run is not None and run.exit_code == 1 and run.kind is CommandKind.MUTATING
    assert tool.runner.calls[0][2] == 120  # default timeout


def test_run_command_timeout_and_wall_clock_cap(repo: Path) -> None:
    tool = Harness(repo, {"sleep 999": Canned(timed_out=True, stdout="partial")})
    result = tool("run_command", {"command": "sleep 999", "timeout_seconds": 5})
    assert not result.ok and result.error_code == "timeout" and "Timed out after 5s" in result.message
    tool.clock.advance(590)  # 10 s left of the 600 s deadline
    tool("run_command", {"command": "echo hi", "timeout_seconds": 300})
    assert tool.runner.calls[-1][2] == pytest.approx(10 - 0.5)


def test_run_command_truncation_is_visible(repo: Path) -> None:
    from agent.host import CapturedStream, ProcessResult

    class Loud(FakeRunner):
        def run(self, command: str, *, cwd: Path, timeout_s: float, max_output_bytes: int) -> ProcessResult:
            return ProcessResult(0, CapturedStream("head\n[... 900 bytes omitted ...]\ntail", 1000, True),
                                 CapturedStream("", 0, False), False, 0.1)

    tool = Harness(repo)
    tool.ctx = ToolContext(tool.ctx.workspace, Loud(tool.clock), tool.ctx.policy, tool.ledger, Limits(),
                           tool.clock, tool.ctx.deadline)
    result = tool("run_command", {"command": "yes | head -c 1000"})
    assert "stdout: 1000 bytes, truncated (start and end kept)" in result.message
    assert "stderr: empty" in result.message


def test_run_command_refused_by_policy_is_not_run(tool: Harness) -> None:
    result = tool("run_command", {"command": "git reset --hard HEAD~3"})
    assert result.error_code == "refused_by_policy"
    assert tool.runner.calls == []
    approval = result.effects.approval
    assert approval is not None and not approval.allowed and approval.kind is CommandKind.DESTRUCTIVE


def test_tool_crash_becomes_internal_error(tool: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: object) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("agent.tools.read_text", boom)
    result = tool("read_file", {"path": "src/calc.py"})
    assert result.error_code == "internal_error" and "disk on fire" in result.message


def test_overlapping_matches_are_ambiguous(tool: Harness, repo: Path) -> None:
    (repo / "braces.txt").write_text("}\n}\n}\n", encoding="utf-8")
    result = tool("edit_file", {"path": "braces.txt", "old_string": "}\n}", "new_string": "]"})
    assert result.error_code == "ambiguous_match" and "lines 1, 2" in result.message


def test_new_files_get_the_umask_mode(tool: Harness, repo: Path) -> None:
    import os

    mask = os.umask(0o022)
    os.umask(mask)
    tool("write_file", {"path": "made.txt", "content": "x\n"})
    assert (repo / "made.txt").stat().st_mode & 0o777 == 0o666 & ~mask


def test_overwrite_requires_having_read_the_whole_file(tool: Harness, repo: Path) -> None:
    (repo / "three.txt").write_text("a\nb\nc\n", encoding="utf-8")
    tool("read_file", {"path": "three.txt", "limit": 1})
    assert tool("write_file", {"path": "three.txt", "content": "new\n"}).error_code == "not_read"
    tool("read_file", {"path": "three.txt", "offset": 2})  # pages 1 and 2-3 together cover the file
    assert tool("write_file", {"path": "three.txt", "content": "new\n"}).ok


def test_search_skips_dangling_links_and_unreadable_files(tool: Harness, repo: Path) -> None:
    import os

    os.symlink(repo / "gone.py", repo / "dangling.py")
    result = tool("search", {"pattern": "def add"})
    assert result.ok and "src/calc.py:1:" in result.data[0].text
    assert "1 file(s) could not be read" in result.message


def test_undecodable_file_names_are_escaped(tool: Harness, repo: Path) -> None:
    import os

    os.close(os.open(os.fsencode(str(repo)) + b"/caf\xe9.txt", os.O_CREAT | os.O_WRONLY, 0o644))
    result = tool("list_files", {"pattern": "*.txt"})
    assert result.ok and result.data[0].text == "caf\\xe9.txt"


def test_an_edit_does_not_license_a_blind_overwrite(tool: Harness, repo: Path) -> None:
    (repo / "three.txt").write_text("a\nb\nc\n", encoding="utf-8")
    tool("read_file", {"path": "three.txt", "limit": 1})
    assert tool("edit_file", {"path": "three.txt", "old_string": "a\n", "new_string": "A\n"}).ok
    assert tool("write_file", {"path": "three.txt", "content": "only\n"}).error_code == "not_read"


def test_a_page_with_cut_lines_does_not_count_as_seen(tool: Harness, repo: Path) -> None:
    (repo / "wide.txt").write_text("x" * 5000 + "\n", encoding="utf-8")
    assert "were cut" in tool("read_file", {"path": "wide.txt"}).message
    assert tool("write_file", {"path": "wide.txt", "content": "short\n"}).error_code == "not_read"


def test_non_utf8_files_are_never_overwritten(tool: Harness, repo: Path) -> None:
    (repo / "latin.txt").write_bytes(b"caf\xe9\n")
    tool("read_file", {"path": "latin.txt"})
    assert tool("write_file", {"path": "latin.txt", "content": "cafe\n"}).error_code == "not_utf8"
    assert (repo / "latin.txt").read_bytes() == b"caf\xe9\n"


def test_command_timeout_default_comes_from_limits(repo: Path) -> None:
    registry = build_registry(Limits(command_timeout_seconds=42, max_command_timeout_seconds=300),
                              allow_delegate=False)
    tool = registry.get("run_command")
    assert tool is not None
    properties: Any = tool.spec.parameters["properties"]
    assert properties["timeout_seconds"]["default"] == 42 and properties["timeout_seconds"]["maximum"] == 300
