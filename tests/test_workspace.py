from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.workspace import PathError, Workspace, atomic_write, compile_glob, split_lines


def test_relative_and_inside_absolute_paths_resolve(repo: Path) -> None:
    ws = Workspace(repo)
    assert ws.resolve("src/calc.py") == repo.resolve() / "src" / "calc.py"
    assert ws.resolve(str(repo / "src" / "calc.py")) == repo.resolve() / "src" / "calc.py"


@pytest.mark.parametrize("raw", ["../outside.txt", "/etc/passwd", "src/../../x", "~/../../etc/hosts"])
def test_escapes_are_refused(repo: Path, raw: str) -> None:
    with pytest.raises(PathError) as info:
        Workspace(repo).resolve(raw)
    assert info.value.code == "path_outside_workspace"


def test_symlink_pointing_outside_is_an_escape(repo: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("key", encoding="utf-8")
    os.symlink(secret, repo / "innocent.txt")
    os.symlink(tmp_path, repo / "linkdir")
    ws = Workspace(repo)
    for raw in ("innocent.txt", "linkdir/secret.txt", "linkdir/new-file.txt"):
        with pytest.raises(PathError) as info:
            ws.resolve(raw, for_write=True)
        assert info.value.code == "path_outside_workspace"


def test_writes_into_git_are_refused_but_reads_allowed(repo: Path) -> None:
    (repo / ".git" / "hooks").mkdir(parents=True)
    ws = Workspace(repo)
    assert ws.resolve(".git/config") == repo.resolve() / ".git" / "config"
    with pytest.raises(PathError) as info:
        ws.resolve(".git/hooks/pre-commit", for_write=True)
    assert info.value.code == "protected_path"


def test_nul_and_empty_paths_are_invalid(repo: Path) -> None:
    for raw in ("a\x00b", "  "):
        with pytest.raises(PathError) as info:
            Workspace(repo).resolve(raw)
        assert info.value.code == "invalid_path"


def test_glob_semantics() -> None:
    def matches(pattern: str, path: str) -> bool:
        return any(p.match(path) for p in compile_glob(pattern))

    assert matches("*.py", "a.py") and not matches("*.py", "pkg/a.py")
    assert matches("**/*.py", "a.py") and matches("**/*.py", "pkg/sub/a.py")
    assert matches("src/**", "src/a/b.txt")
    assert matches("**/test_*.{py,txt}", "t/test_x.txt") and not matches("**/test_*.{py,txt}", "t/test_x.md")
    assert matches("file?.[ch]", "file1.c") and not matches("file?.[ch]", "file12.c")
    with pytest.raises(PathError):
        compile_glob("../*.py")


def test_atomic_write_preserves_mode_and_bytes(tmp_path: Path) -> None:
    target = tmp_path / "run.sh"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o755)
    atomic_write(target, "line1\r\nline2\n")
    assert target.read_bytes() == b"line1\r\nline2\n"
    assert target.stat().st_mode & 0o777 == 0o755
    assert [p.name for p in tmp_path.iterdir()] == ["run.sh"]  # no temp file left behind


def test_split_lines_only_on_newline() -> None:
    assert split_lines("a\fb\r\nc\n") == ["a\fb", "c"]


@pytest.mark.parametrize("raw", [".GIT/hooks/pre-commit", ".Git/config", "vendor/lib/.git/hooks/post-checkout"])
def test_git_directories_are_protected_case_insensitively_at_any_depth(repo: Path, raw: str) -> None:
    with pytest.raises(PathError) as info:
        Workspace(repo).resolve(raw, for_write=True)
    assert info.value.code == "protected_path"
