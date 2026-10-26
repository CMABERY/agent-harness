"""The workspace boundary: which paths the file tools may touch, and how.

Decisions owned here:
  * Every path is resolved (symlinks followed) and must land inside the root.
    A symlink inside the repo that points outside is an escape, not a path.
  * ``~`` is never expanded and absolute paths are accepted only if they
    resolve inside the root, so "/etc/passwd" and "../../x" fail the same way.
  * Writes under any ``.git`` directory (compared case-insensitively, for
    case-insensitive filesystems) are refused: a hook written there executes
    on the user's next commit, outside every control this harness has.
  * Writes are atomic (temp file + rename) and preserve the file mode, so an
    interrupted run never leaves a half-written source file.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
    }
)
PROTECTED_DIRS = frozenset({".git"})  # compared case-folded
MAX_READ_BYTES = 10_000_000
_BINARY_SNIFF_BYTES = 8192


def _umask() -> int:
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


# mkstemp creates files 0600; a new source file should get the mode `touch` would give it.
_NEW_FILE_MODE = 0o666 & ~_umask()


class PathError(Exception):
    """A path the model supplied cannot be used. ``code`` is shown to the model."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Workspace:
    def __init__(self, root: Path) -> None:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(f"workspace root is not a directory: {root}")
        self.root = resolved

    def resolve(self, raw: str, *, for_write: bool = False) -> Path:
        if "\x00" in raw:
            raise PathError("invalid_path", "path contains a NUL byte")
        if not raw.strip():
            raise PathError("invalid_path", "path is empty")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = Path(os.path.realpath(candidate))
        if not resolved.is_relative_to(self.root):
            raise PathError(
                "path_outside_workspace",
                f"{raw!r} resolves outside the workspace root, so it was refused. "
                "File tools can only access paths inside the workspace; use a path "
                "relative to the workspace root.",
            )
        if for_write:
            parts = resolved.relative_to(self.root).parts
            protected = next((p for p in parts if p.casefold() in PROTECTED_DIRS), None)
            if protected is not None:
                raise PathError(
                    "protected_path",
                    f"{raw!r} is inside {protected}/, which file tools never write to.",
                )
        return resolved

    def relative(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return rel if rel else "."

    def contains(self, path: Path) -> bool:
        return Path(os.path.realpath(path)).is_relative_to(self.root)


# --------------------------------------------------------------------------- #
# Globbing
# --------------------------------------------------------------------------- #


def expand_braces(pattern: str) -> list[str]:
    """Expand one level of ``{a,b}`` alternatives (recursively for later groups)."""
    start = pattern.find("{")
    end = pattern.find("}", start + 1) if start != -1 else -1
    if start == -1 or end == -1:
        return [pattern]
    head, body, tail = pattern[:start], pattern[start + 1 : end], pattern[end + 1 :]
    expanded: list[str] = []
    for option in body.split(","):
        expanded.extend(expand_braces(head + option + tail))
    return expanded


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob (``*``, ``**``, ``?``, ``[...]``) to an anchored regex.

    ``*`` never crosses ``/``; ``**/`` matches zero or more directories.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and (close := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1 : close].replace("\\", "\\\\")
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append(f"[{body}]")
            i = close + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def compile_glob(pattern: str) -> list[re.Pattern[str]]:
    if pattern.startswith("/") or ".." in Path(pattern).parts:
        raise PathError("invalid_pattern", "glob patterns must be relative and must not contain '..'")
    return [glob_to_regex(p) for p in expand_braces(pattern)]


def walk_files(workspace: Workspace, base: Path, patterns: list[re.Pattern[str]]) -> Iterator[Path]:
    """Yield files under ``base`` whose base-relative path matches, in sorted order.

    Ignored directories are pruned before descent (so node_modules costs
    nothing), directory symlinks are not followed, and file symlinks that
    resolve outside the workspace are skipped.
    """
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        current = Path(dirpath)
        for name in sorted(filenames):
            path = current / name
            rel = path.relative_to(base).as_posix()
            if any(p.match(rel) for p in patterns) and workspace.contains(path):
                yield path


# --------------------------------------------------------------------------- #
# Reading and writing text
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TextFile:
    text: str
    valid_utf8: bool
    binary: bool
    size: int


def read_text(path: Path) -> TextFile:
    """Read a file as text without ever raising on its encoding."""
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        raise PathError(
            "file_too_large",
            f"file is {size} bytes (limit {MAX_READ_BYTES}); use search, or run_command "
            "with head/sed to read a slice",
        )
    raw = path.read_bytes()
    if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
        return TextFile(text="", valid_utf8=False, binary=True, size=size)
    try:
        return TextFile(text=raw.decode("utf-8"), valid_utf8=True, binary=False, size=size)
    except UnicodeDecodeError:
        return TextFile(text=raw.decode("utf-8", errors="replace"), valid_utf8=False, binary=False, size=size)


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` exactly (no newline translation) via temp file + rename.

    An existing file keeps its mode; a new one gets 0666 minus the umask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else _NEW_FILE_MODE
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def split_lines(text: str) -> list[str]:
    """Split on ``\\n`` only (``str.splitlines`` also splits on form feeds and
    friends, which would make line numbers disagree with grep and editors).
    A trailing ``\\r`` is dropped from each line for display."""
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]


def count_lines(text: str) -> int:
    return len(split_lines(text))
