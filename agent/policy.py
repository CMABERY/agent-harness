"""Command classification and approval.

Decisions owned here:
  * Classification is lexical, not semantic. It guards against a model that
    *means well but errs* or is being steered by injected text; it is not a
    sandbox, and DESIGN.md says so.
  * Fail closed on syntax the reviewer does not model. A command that cannot
    be scanned (unbalanced quotes, ANSI-C ``$'...'`` quoting), whose program
    name is computed at run time, or which passes computed arguments to a
    program whose classification depends on its arguments (``rm $FLAGS x``,
    ``git $(echo reset) --hard``) is DESTRUCTIVE, so the default policy refuses
    it and tells the model to write the command literally.
  * A recursive, quote-aware scanner runs before ``shlex``. It treats ``#`` as
    a comment only at the start of a word; turns unquoted newlines into
    separators; reads heredoc delimiters with shell quote removal; treats a
    heredoc body as data unless its line runs a shell (then it is code) or its
    delimiter is unquoted (then its substitutions run); scans ``$(...)``,
    ``${...}``, ``$((...))``, backticks and ``<(...)`` to their real end
    (heredocs and quotes inside included); and classifies every command
    substitution as a command in its own right. Fused operator tokens such as
    ``);`` or ``&&(`` are split before segmenting.
  * Wrappers (sudo, env, nice, timeout, xargs, ``sh -c``, ``bash -lc``, eval,
    find -exec) are unwrapped; shell keywords and ``function NAME`` are
    stripped. Long options match by unambiguous prefix, as git and GNU tools
    accept them (``git reset --har``).
  * Every destructive part is reviewed separately, and the operator confirms
    each one. An allowed recursive delete never approves anything else.
  * ``masks_exit_status`` marks commands whose exit status may not reflect
    their checks. Under ``bash -e -o pipefail`` the status is decided by the
    last top-level statement, or by an earlier one that fails. So the command
    is masked when its last statement is masked (``tests || true``,
    ``! tests``), when the last statement is read-only and an earlier one was
    masked or errexit was switched off (``tests || true; echo done``,
    ``set +e; tests; echo done``), or when pipefail was switched off and the
    last statement is a pipeline. ``pip install -e . || true; pytest -q`` is
    not masked: pytest decides its status. The gate never accepts a masked
    command as a passing check.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from agent.types import CommandKind

_SEVERITY = {CommandKind.READ_ONLY: 0, CommandKind.MUTATING: 1, CommandKind.DESTRUCTIVE: 2}
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "|&", "(", ")", ";;"})
_OPERATORS = ("&>>", "<<<", "&&", "||", ";;", "|&", "&>", ">>", ">&", ">|", "<<", "<&", "<>",
              ";", "&", "|", "(", ")", "<", ">")
_PUNCTUATION = frozenset("();<>|&")
_WRITE_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", ">&"})
_READ_REDIRECTS = frozenset({"<", "<<", "<<<", "<&", "<>"})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_KEYWORDS = frozenset({"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "{", "}", "!", "esac"})
_SUBST = "__substitution__"  # stands in for the output of $(...) and backticks
_EXPAND = "__expansion__"  # stands in for $VAR, ${...} and $((...))
_PLACEHOLDERS = (_SUBST, _EXPAND)

_READ_ONLY = frozenset(
    {
        "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "ag", "pwd",
        "echo", "printf", "true", "false", "which", "type", "file", "stat", "du", "df",
        "tree", "uniq", "cut", "tr", "diff", "cmp", "basename", "dirname", "realpath",
        "readlink", "date", "whoami", "id", "uname", "printenv", "sha256sum", "sha1sum",
        "md5sum", "nl", "jq", "column", "xxd", "od", "hexdump", "test", "[", "cd", "sleep",
    }
)
# Wrappers run another command; each set names the wrapper's options that take a separate value.
_WRAPPERS: dict[str, frozenset[str]] = {
    "env": frozenset({"-u", "--unset", "-C", "--chdir"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "ionice": frozenset({"-c", "-n", "-p", "--class", "--classdata"}),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "xargs": frozenset({"-I", "-J", "-n", "-P", "-L", "-d", "-E", "-s", "-a", "-R", "-S",
                        "--max-args", "--max-procs", "--delimiter", "--arg-file"}),
    "exec": frozenset({"-a"}),
    "nohup": frozenset(),
    "command": frozenset(),
    "builtin": frozenset(),
}
_PRIVILEGED = frozenset({"sudo", "doas", "su", "pkexec"})
_DISK = frozenset({"shred", "wipefs", "fdisk", "sfdisk", "parted"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
_CODE_CONSUMERS = _SHELLS | {"eval", "source", ".", "ssh", "xargs"}
_SHELL_VALUE_OPTIONS = frozenset({"-o", "+o", "-O", "+O", "--rcfile", "--init-file"})
_DIR_CHANGERS = frozenset({"cd", "pushd", "popd"})
_COMPUTED_ARGS_DESTRUCTIVE = frozenset({"rm", "find", "dd", "eval"})  # a computed word could be -rf, -delete, of=
# Options whose next word is a value, so a computed word there cannot become a flag (`git commit -m "$(cat ...)"`).
_GIT_MESSAGE_OPTIONS = frozenset({"-m", "--message", "-F", "--file", "--author", "--date", "-t", "--template",
                                  "--trailer", "-C", "-c", "--reuse-message", "--reedit-message"})
_GIT_VALUE_OPTIONS = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix",
                                "--config-env"})
_GIT_READ_ONLY = frozenset(
    {
        "status", "diff", "log", "show", "blame", "grep", "ls-files", "ls-tree",
        "rev-parse", "describe", "shortlog", "cat-file", "show-ref", "version", "help",
    }
)


@dataclass(frozen=True, slots=True)
class Classification:
    kind: CommandKind
    reason: str
    category: str = ""  # recursive_delete(_unresolved) | git_destructive | privileged | disk | unparseable | computed_program
    delete_targets: tuple[str, ...] = ()
    masks_exit_status: bool = False  # set on the whole-command classification only


@dataclass(frozen=True, slots=True)
class Review:
    classification: Classification
    allowed: bool
    reason: str


class ApprovalPolicy(Protocol):
    def review(self, command: str, cwd: Path) -> Review: ...


Confirm = Callable[[str, Classification], bool]

_READ = Classification(CommandKind.READ_ONLY, "known read-only command")


def _mutating(reason: str) -> Classification:
    return Classification(CommandKind.MUTATING, reason)


def _destructive(reason: str, category: str, targets: Sequence[str] = ()) -> Classification:
    return Classification(CommandKind.DESTRUCTIVE, reason, category, tuple(targets))


def _worst(items: Sequence[Classification]) -> Classification:
    worst = _READ
    for item in items:
        if _SEVERITY[item.kind] > _SEVERITY[worst.kind]:
            worst = item
    return worst


def _computed(word: str) -> bool:
    return any(p in word for p in _PLACEHOLDERS)


def _long(arg: str, option: str) -> bool:
    """``arg`` is ``option``, ``option=value``, or an abbreviation that the tool would expand to it."""
    return arg == option or arg.startswith(option + "=") or (
        arg.startswith("--") and len(arg) >= 3 and "=" not in arg and option.startswith(arg))


def _has(args: Sequence[str], *options: str) -> bool:
    return any(_long(a, o) for a in args for o in options)


# --------------------------------------------------------------------------- #
# Scanning: the shell syntax shlex does not understand
# --------------------------------------------------------------------------- #


class _ScanError(ValueError):
    pass


def _closing_backtick(text: str, start: int) -> int:
    i = start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "`":
            return i
        i += 1
    raise _ScanError("unbalanced backtick")


def _heredoc_word(text: str, start: int) -> tuple[str, bool, int]:
    """Read a heredoc delimiter with shell quote removal: (word, was any part quoted, end)."""
    word: list[str] = []
    quoted = False
    quote = ""
    i = start
    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                quote = ""
            else:
                word.append(c)
        elif c in "'\"":
            quote, quoted = c, True
        elif c == "\\" and i + 1 < len(text):
            word.append(text[i + 1])
            quoted = True
            i += 1
        elif c.isspace() or c in _PUNCTUATION:
            break
        else:
            word.append(c)
        i += 1
    if quote or not word:
        raise _ScanError("malformed heredoc delimiter")
    return "".join(word), quoted, i


def _heredoc_body(text: str, start: int, delimiter: str, strip_tabs: bool) -> tuple[str, int]:
    lines: list[str] = []
    i = start
    while i < len(text):
        end = text.find("\n", i)
        line = text[i:] if end == -1 else text[i:end]
        i = len(text) if end == -1 else end + 1
        if (line.lstrip("\t") if strip_tabs else line) == delimiter:
            return "\n".join(lines), i
        lines.append(line)
    return "\n".join(lines), i


class _Scanner:
    """Rewrites a command for shlex and collects the commands nested inside it."""

    def __init__(self, text: str) -> None:
        self.text = text

    def top_level(self) -> tuple[str, list[str]]:
        nested: list[str] = []
        rewritten, _ = self._scan(0, "", nested)
        return rewritten, nested

    def expanded_body(self, nested: list[str]) -> None:
        """Collect the substitutions in an unquoted heredoc body, where only ``$`` and backticks are special."""
        i = 0
        while i < len(self.text):
            if self.text[i] == "\\":
                i += 2
            elif self.text[i] in "$`":
                i = self._expansion(i, [], nested)
            else:
                i += 1

    def _scan(self, i: int, closer: str, nested: list[str]) -> tuple[str, int]:
        """Rewrite from ``i``; with a closer (")" or "}"), stop after the matching one."""
        text, n = self.text, len(self.text)
        out: list[str] = []
        pending: list[tuple[str, bool, bool]] = []  # heredocs: (delimiter, strip tabs, body is expanded)
        line_start = 0
        quote = ""
        word_start = True
        depth = 0
        while i < n:
            c = text[i]
            if quote == "'":
                out.append(c)
                quote = "" if c == "'" else quote
                i += 1
                continue
            if quote == '"':
                if c == "\\":
                    out.append(text[i : i + 2])
                    i += 2
                elif c in "$`":
                    i = self._expansion(i, out, nested)
                else:
                    out.append(c)
                    quote = "" if c == '"' else quote
                    i += 1
                continue
            if closer and c == closer and depth == 0:
                return "".join(out), i + 1
            if c == "\\":
                if text.startswith("\\\n", i):  # line continuation
                    i += 2
                    continue
                out.append(text[i : i + 2])
                i += 2
                word_start = False
                continue
            if c in "'\"":
                quote = c
                out.append(c)
                i += 1
                word_start = False
                continue
            if c == "#" and word_start and closer != "}":  # inside ${...}, '#' is an operator
                end = text.find("\n", i)
                i = n if end == -1 else end
                continue
            if c == "\n":
                line_words = "".join(out[line_start:]).split()
                is_code = any(os.path.basename(w) in _CODE_CONSUMERS for w in line_words)
                out.append(" ; ")
                i += 1
                for delimiter, strip_tabs, expands in pending:
                    body, i = _heredoc_body(text, i, delimiter, strip_tabs)
                    if is_code:
                        nested.append(body)  # `bash <<EOF`, `cat <<EOF | sh`: the body runs
                    elif expands:
                        _Scanner(body).expanded_body(nested)  # `cat <<EOF` still runs its $(...)
                pending.clear()
                line_start = len(out)
                word_start = True
                continue
            if c in "$`":
                if text.startswith("$'", i):
                    raise _ScanError("ANSI-C quoting ($'...') is not supported by the reviewer; use plain quotes")
                if text.startswith('$"', i):  # locale quoting behaves like "..."
                    i += 1
                    continue
                i = self._expansion(i, out, nested)
                word_start = False
                continue
            if text.startswith(("<(", ">("), i):
                end = self._scan(i + 2, ")", [])[1]
                nested.append(text[i + 2 : end - 1])
                out.append(_SUBST)
                i = end
                word_start = False
                continue
            if text.startswith("<<<", i):  # a here-string, not a heredoc
                out.append(" <<< ")
                i += 3
                word_start = True
                continue
            if text.startswith("<<", i):
                j = i + 2
                strip_tabs = j < n and text[j] == "-"
                j += 1 if strip_tabs else 0
                while j < n and text[j] in " \t":
                    j += 1
                delimiter, quoted, i = _heredoc_word(text, j)
                pending.append((delimiter, strip_tabs, not quoted))
                out.append(" << heredoc ")
                word_start = True
                continue
            if closer == ")" and c == "(":
                depth += 1
            elif closer == ")" and c == ")":
                depth -= 1
            elif closer == "}" and c == "{":
                depth += 1
            elif closer == "}" and c == "}":
                depth -= 1
            out.append(c)
            word_start = c in " \t" or c in _PUNCTUATION
            i += 1
        if quote:
            raise _ScanError("unbalanced quote")
        if closer:
            raise _ScanError(f"missing closing '{closer}'")
        return "".join(out), i

    def _expansion(self, i: int, out: list[str], nested: list[str]) -> int:
        """Handle one ``$...`` or backtick construct at ``i``; return the index after it."""
        text = self.text
        if text.startswith("$((", i):  # arithmetic; may contain substitutions
            _, end = self._scan(i + 3, ")", nested)
            if not text.startswith(")", end):
                raise _ScanError("unbalanced arithmetic expansion")
            out.append(_EXPAND)
            return end + 1
        if text.startswith("$(", i):
            _, end = self._scan(i + 2, ")", [])  # the inner text is classified on its own below
            nested.append(text[i + 2 : end - 1])
            out.append(_SUBST)
            return end
        if text.startswith("${", i):
            _, end = self._scan(i + 2, "}", nested)
            out.append(_EXPAND)
            return end
        if text[i] == "`":
            end = _closing_backtick(text, i)
            nested.append(text[i + 1 : end])
            out.append(_SUBST)
            return end + 1
        match = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*|[0-9@*#?$!-])", text[i:])
        if match:
            out.append(_EXPAND)
            return i + match.end()
        out.append("$")
        return i + 1


def _tokens(text: str) -> list[str]:
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens: list[str] = []
    for token in lexer:
        if token and set(token) <= _PUNCTUATION:  # shlex fuses runs like ");" or "&&("
            position = 0
            while position < len(token):
                operator = next((op for op in _OPERATORS if token.startswith(op, position)), None)
                if operator is None:
                    raise _ScanError(f"unknown operator {token!r}")
                tokens.append(operator)
                position += len(operator)
        else:
            tokens.append(token)
    return tokens


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Analysis:
    parts: list[Classification]
    masks_exit_status: bool


def _analyze(command: str, depth: int) -> _Analysis:
    if depth > 4:
        return _Analysis([_destructive("command nests too deeply to review", "unparseable")], False)
    try:
        rewritten, nested = _Scanner(command).top_level()
        tokens = _tokens(rewritten)
    except ValueError as exc:
        return _Analysis([_destructive(f"command could not be parsed for review ({exc})", "unparseable")], False)
    segments: list[tuple[str, list[str]]] = [("", [])]  # (separator before it, tokens)
    for token in tokens:
        if token in _SEPARATORS:
            segments.append((token, []))
        else:
            segments[-1][1].append(token)
    parts: list[Classification] = []
    statements: list[_Statement] = [_Statement()]  # top-level statements, split at ; & and newlines
    nesting = 0
    errexit_off = pipefail_off = False
    moved = False  # after `cd`, relative delete targets no longer mean what the policy would resolve
    previous: list[str] = []
    for separator, segment in segments:
        nesting += separator == "("
        nesting -= separator == ")"
        if separator in {";", "&"} and nesting <= 0 and statements[-1].used:
            statements[-1].errexit_off_before, statements[-1].pipefail_off_before = errexit_off, pipefail_off
            statements.append(_Statement())
        current = statements[-1]
        current.used = current.used or bool(segment)
        current.masked = current.masked or separator == "||" or segment[:1] == ["!"]
        current.piped = current.piped or separator in {"|", "|&"}
        if segment[:1] == ["function"]:
            segment = segment[2:]
        while segment and segment[0] in _KEYWORDS:
            segment = segment[1:]
        if not segment:
            previous = []
            continue
        if segment[0] == "set":
            switched = " ".join(segment[1:])
            errexit_off = errexit_off or bool(re.search(r"(^|\s)\+[a-zA-Z]*e|\+o errexit", switched))
            pipefail_off = pipefail_off or "+o pipefail" in switched
        found = _classify_simple(segment, depth)
        current.checks = current.checks or found.kind is not CommandKind.READ_ONLY
        if separator in {"|", "|&"} and _reads_code_from_stdin(segment):
            found = _worst([found, _piped_code(previous, depth)])
        if moved and found.category == "recursive_delete":
            found = _destructive("recursive delete after a directory change; its targets cannot be resolved",
                                 "recursive_delete_unresolved")
        moved = moved or os.path.basename(segment[0]) in _DIR_CHANGERS
        parts.append(found)
        previous = segment
    statements[-1].errexit_off_before, statements[-1].pipefail_off_before = errexit_off, pipefail_off
    for inner in nested:
        parts.extend(_analyze(inner, depth + 1).parts)
    return _Analysis(parts, _masks_exit_status([s for s in statements if s.used]))


@dataclass(slots=True)
class _Statement:
    used: bool = False
    masked: bool = False  # contains `||` or starts with `!`
    piped: bool = False
    checks: bool = False  # contains a command that is not read-only
    errexit_off_before: bool = False
    pipefail_off_before: bool = False


def _masks_exit_status(statements: list[_Statement]) -> bool:
    if not statements:
        return False
    last, earlier = statements[-1], statements[:-1]
    if last.masked or (last.piped and last.pipefail_off_before):
        return True
    # A read-only last statement reports success unless an earlier statement failed and stopped the script;
    # that only holds while errexit is on and no earlier failure was swallowed.
    return not last.checks and (any(s.masked for s in earlier) or last.errexit_off_before)


def classify_parts(command: str) -> list[Classification]:
    """Classify every simple command that ``command`` would run, nested ones included."""
    return _analyze(command, 0).parts


def classify(command: str) -> Classification:
    analysis = _analyze(command, 0)
    return replace(_worst(analysis.parts), masks_exit_status=analysis.masks_exit_status)


def _unwrap(words: list[str]) -> list[str]:
    """Strip leading assignments and wrapper programs with their own options."""
    while words:
        while words and _ASSIGNMENT.match(words[0]):
            words = words[1:]
        program = os.path.basename(words[0]) if words else ""
        if program not in _WRAPPERS:
            return words
        words = _without_options(words[1:], _WRAPPERS[program], assignments=program == "env")
        if program == "timeout":
            words = words[1:]  # the duration
    return words


def _reads_code_from_stdin(words: list[str]) -> bool:
    """A shell with no -c and no script file executes whatever arrives on stdin."""
    words = _unwrap(words)
    program = os.path.basename(words[0]) if words else ""
    if program not in _SHELLS:
        return False
    args = words[1:]
    if _shell_command_string(args) is not None:
        return False
    index = 0
    while index < len(args) and args[index].startswith(("-", "+")):
        if "s" in args[index][1:] and not args[index].startswith("--"):
            return True  # sh -s: read commands from stdin even with arguments
        index += 2 if args[index] in _SHELL_VALUE_OPTIONS else 1
    return index >= len(args)


def _piped_code(producer: list[str], depth: int) -> Classification:
    """What a shell will run when it reads a pipe: reviewable only for a literal echo or a plain printf."""
    program = os.path.basename(producer[0]) if producer else ""
    args = producer[1:]
    payload: str | None = None
    if program == "echo":
        index = 0
        while index < len(args) and re.fullmatch(r"-[neE]+", args[index]):
            index += 1  # echo's own options; everything after them is the payload, flags included
        payload = " ".join(args[index:])
    elif program == "printf" and len(args) == 1 and "%" not in args[0]:
        payload = args[0]
    if payload is None or _computed(payload) or "\\" in payload:
        # Unknown producers, computed text, and escapes that some shells' echo expands (`\n`) cannot be reviewed.
        return _destructive("pipes the output of another program into a shell, so what runs cannot be reviewed",
                            "unparseable")
    return _worst(_analyze(payload, depth + 1).parts)


def _classify_simple(tokens: list[str], depth: int) -> Classification:
    words: list[str] = []
    herestrings: list[str] = []
    writes_file = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _WRITE_REDIRECTS or token in _READ_REDIRECTS:
            if words and words[-1].isdigit():
                words.pop()  # the file-descriptor number of "2>file"
            target = tokens[index + 1] if index + 1 < len(tokens) else ""
            if token == "<<<":
                herestrings.append(target)
            is_dup = token == ">&" and target.isdigit()
            if token in _WRITE_REDIRECTS and target != "/dev/null" and not is_dup:
                writes_file = True
            index += 2
            continue
        words.append(token)
        index += 1
    while words and _ASSIGNMENT.match(words[0]):
        words.pop(0)
    base = _classify_words(words, depth) if words else _READ
    if herestrings and _reads_code_from_stdin(words):  # bash <<< 'rm -rf ~'
        base = _worst([base, *(c for h in herestrings for c in _analyze(h, depth + 1).parts)])
    if writes_file and base.kind is CommandKind.READ_ONLY:
        return _mutating("redirects output into a file")
    return base


def _without_options(args: list[str], takes_value: frozenset[str], assignments: bool = False) -> list[str]:
    """Strip a wrapper's own options so the wrapped command is classified intact."""
    index = 0
    while index < len(args):
        arg = args[index]
        if assignments and _ASSIGNMENT.match(arg):
            index += 1
        elif arg == "--":
            return args[index + 1 :]
        elif arg.startswith("-"):
            index += 2 if arg in takes_value else 1
        else:
            break
    return args[index:]


def _shell_command_string(args: list[str]) -> str | None:
    """The script of ``sh -c SCRIPT``, ``bash -lc SCRIPT``, ``bash -o pipefail -c SCRIPT``."""
    index, has_c = 0, False
    while index < len(args):
        arg = args[index]
        if arg in _SHELL_VALUE_OPTIONS:
            index += 2
        elif arg == "--":
            index += 1
            break
        elif arg.startswith(("-", "+")):
            has_c = has_c or (not arg.startswith("--") and "c" in arg[1:])
            index += 1
        else:
            break
    return args[index] if has_c and index < len(args) else None


def _classify_words(words: list[str], depth: int) -> Classification:
    if _computed(words[0]) or "$" in words[0]:
        return _destructive("the program name is computed at run time, so the command cannot be reviewed; "
                            "write the program name literally", "computed_program")
    program = os.path.basename(words[0])
    args = words[1:]
    if program in _PRIVILEGED:
        return _destructive(f"runs a command with elevated privileges ({program})", "privileged")
    if program in _WRAPPERS:
        return _classify_wrapper(program, args, depth)
    if program in _COMPUTED_ARGS_DESTRUCTIVE and any(map(_computed, args)):
        return _computed_arguments(program)
    if program in _SHELLS:
        script = _shell_command_string(args)
        if script is not None:
            return _computed_arguments(program) if _computed(script) else _worst(_analyze(script, depth + 1).parts)
        if any(map(_computed, args)):  # bash <(curl ...): a script nobody can read first
            return _computed_arguments(program)
        return _mutating(f"{program} runs a script")
    if program == "eval":
        return _worst(_analyze(" ".join(args), depth + 1).parts)
    if program in {"source", "."}:
        return _computed_arguments(program) if any(map(_computed, args)) else _mutating(f"{program} runs a script")
    if program == "rm":
        return _classify_rm(args)
    if program == "find":
        return _classify_find(args, depth)
    if program == "git":
        return _classify_git(args)
    if program == "dd":
        if any(a.startswith("of=") for a in args):
            return _destructive("dd writes raw bytes to its output target", "disk")
        return _mutating("dd")
    if program in _DISK or program.startswith("mkfs"):
        return _destructive(f"{program} destroys data on a device or file", "disk")
    if program == "sed":
        in_place = any(_long(a, "--in-place") or (a.startswith("-") and not a.startswith("--") and "i" in a)
                       or _computed(a) for a in args)
        return _mutating("sed -i edits files in place") if in_place else _READ
    if program == "sort":
        writes = any(a.startswith("-o") or _long(a, "--output") or _computed(a) for a in args)
        return _mutating("sort -o writes a file") if writes else _READ
    if program in _READ_ONLY:
        return _READ
    # An unknown program may be a wrapper (setsid, flock, chrt, docker run ...): if what follows its own
    # options is itself a destructive command line, treat the whole thing as that command.
    embedded = _without_options(args, frozenset())
    if embedded and depth < 8:
        inner = _classify_words(embedded, depth + 1)
        # A computed word after an unknown program is far more often data (`pytest $(git diff ...)`) than a
        # command, so only a concrete destructive command line is propagated.
        if inner.kind is CommandKind.DESTRUCTIVE and inner.category != "computed_program":
            return replace(inner, reason=f"{program} may run it: {inner.reason}")
    return _mutating(f"{program} is not known to be read-only")


def _computed_arguments(program: str) -> Classification:
    return _destructive(f"{program} gets arguments that are computed at run time ($VAR or $(...)), so its effect "
                        "cannot be reviewed; write the values literally", "unparseable")


def _classify_wrapper(program: str, args: list[str], depth: int) -> Classification:
    if program == "env":
        for position, arg in enumerate(args):
            if _long(arg, "--split-string") or arg == "-S":
                following = args[position + 1] if position + 1 < len(args) else ""
                return _worst(_analyze(arg.split("=", 1)[1] if "=" in arg else following, depth + 1).parts)
            if arg.startswith("-S") and len(arg) > 2:
                return _worst(_analyze(arg[2:], depth + 1).parts)
    rest = _without_options(args, _WRAPPERS[program], assignments=program == "env")
    if program == "timeout":
        rest = rest[1:]  # the duration
    if not rest:
        return _READ
    inner = _classify_words(rest, depth)
    options = args[: len(args) - len(rest)]
    changes_dir = program == "env" and any(a in {"-C", "--chdir"} or a.startswith(("--chdir=", "-C"))
                                           for a in options)
    if changes_dir and inner.category == "recursive_delete":
        return _destructive("recursive delete in another directory (env -C); its targets cannot be resolved",
                            "recursive_delete_unresolved")
    return inner


def _classify_rm(args: list[str]) -> Classification:
    recursive = False
    targets: list[str] = []
    options_done = False
    for arg in args:
        if not options_done and arg == "--":
            options_done = True
        elif not options_done and arg.startswith("--"):
            recursive = recursive or _long(arg, "--recursive")
        elif not options_done and arg.startswith("-") and len(arg) > 1:
            recursive = recursive or "r" in arg or "R" in arg
        else:
            targets.append(arg)
    if recursive:
        return _destructive("recursive delete", "recursive_delete", targets)
    return _mutating("deletes files")


def _classify_find(args: list[str], depth: int) -> Classification:
    starts: list[str] = []
    for arg in args:
        if arg.startswith("-") or arg in {"(", "!"}:
            break
        starts.append(arg)
    targets = starts or ["."]
    if "-delete" in args:
        return _destructive("find -delete removes every matching file", "recursive_delete", targets)
    for flag in ("-exec", "-execdir", "-ok", "-okdir"):
        if flag in args:
            inner = args[args.index(flag) + 1 :]
            inner = inner[: inner.index(";")] if ";" in inner else inner
            if inner and os.path.basename(inner[0]) == "rm":
                return _destructive("find -exec rm deletes matching files", "recursive_delete", targets)
            return _worst([_classify_words(inner, depth), _mutating("find -exec")]) if inner else _READ
    return _READ


def _classify_git(args: list[str]) -> Classification:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        index += 2 if args[index] in _GIT_VALUE_OPTIONS else 1
    if index >= len(args):
        return _READ
    sub, rest = args[index], args[index + 1 :]
    if any(_computed(a) for a in args[:index + 1]) or any(
        _computed(a) and (position == 0 or rest[position - 1] not in _GIT_MESSAGE_OPTIONS)
        for position, a in enumerate(rest)
    ):
        return _computed_arguments("git")
    flags = set(rest)
    short = "".join(a[1:] for a in rest if a.startswith("-") and not a.startswith("--"))
    deleting = "d" in short or _has(rest, "--delete")
    forcing = "f" in short or _has(rest, "--force")

    def destructive(why: str) -> Classification:
        return _destructive(f"git {sub}: {why}", "git_destructive")

    if sub == "reset" and _has(rest, "--hard"):
        return destructive("--hard discards uncommitted work")
    if sub == "clean" and forcing:
        return destructive("deletes untracked files")
    if sub == "push" and (
        forcing or deleting or _has(rest, "--force-with-lease", "--force-if-includes", "--mirror", "--prune")
        or any(a.startswith("+") or (a.startswith(":") and len(a) > 1) for a in rest)
    ):
        return destructive("rewrites or deletes remote history")
    if sub == "checkout" and ("--" in flags or "." in flags or forcing):
        return destructive("discards working-tree changes")
    if sub == "restore" and (not (_has(rest, "--staged") or "S" in short) or _has(rest, "--worktree") or "W" in short):
        return destructive("discards working-tree changes")
    if sub == "branch" and ("D" in short or (deleting and forcing)):
        return destructive("force-deletes a branch")
    if sub == "switch" and (_has(rest, "--discard-changes") or forcing):
        return destructive("discards working-tree changes")
    if sub == "stash" and rest[:1] in (["drop"], ["clear"]):
        return destructive("deletes stashed work")
    if sub == "reflog" and rest[:1] in (["expire"], ["delete"]):
        return destructive("destroys recovery history")
    if sub in {"filter-branch", "filter-repo"}:
        return destructive("rewrites history")
    if sub == "update-ref" and "d" in short:
        return destructive("deletes a ref")
    if sub == "gc" and _has(rest, "--prune"):
        return destructive("prunes unreachable objects")
    if sub == "rm" and forcing:
        return destructive("force-removes files, including uncommitted changes")
    if sub in _GIT_READ_ONLY:
        return _READ
    listing = {"-a", "-r", "-v", "-vv", "--all", "--remotes", "--list", "--show-current"}
    if sub == "branch" and all(a in listing for a in rest):
        return _READ
    return _mutating(f"git {sub} changes repository state")


class DefaultPolicy:
    def __init__(
        self,
        workspace_root: Path,
        allowed_delete_paths: Sequence[str] = (),
        confirm: Confirm | None = None,
    ) -> None:
        self._root = workspace_root.resolve()
        self._allowed = tuple(Path(os.path.realpath(self._root / p)) for p in allowed_delete_paths)
        self._confirm = confirm

    def review(self, command: str, cwd: Path) -> Review:
        analysis = _analyze(command, 0)
        found = replace(_worst(analysis.parts), masks_exit_status=analysis.masks_exit_status)
        if found.kind is not CommandKind.DESTRUCTIVE:
            return Review(found, True, f"allowed ({found.kind.value})")
        blocking = [
            p for p in analysis.parts
            if p.kind is CommandKind.DESTRUCTIVE
            and not (p.category == "recursive_delete" and self._targets_allowed(p.delete_targets, cwd))
        ]
        if not blocking:
            return Review(found, True, "recursive delete confined to explicitly allowed paths")
        masked = found.masks_exit_status
        if self._confirm is None:
            return Review(replace(blocking[0], masks_exit_status=masked), False,
                          f"refused by the default policy: {blocking[0].reason}")
        for part in blocking:  # the operator decides every blocking part, not just the first
            if not self._confirm(command, part):
                return Review(replace(part, masks_exit_status=masked), False, f"declined by the operator: {part.reason}")
        return Review(found, True, "approved by the operator")

    def _targets_allowed(self, targets: Sequence[str], cwd: Path) -> bool:
        if not targets or not self._allowed:
            return False
        for target in targets:
            if target.startswith("~") or "$" in target or _computed(target) or "{" in target or "}" in target:
                return False  # cannot know what the shell will expand this to
            glob_at = min((i for i, ch in enumerate(target) if ch in "*?["), default=-1)
            if glob_at != -1 and ".." in Path(target).parts:
                return False  # build/*/../../src: the shell resolves that glob, not us
            literal = target if glob_at == -1 else target[: target.rfind("/", 0, glob_at) + 1]
            resolved = Path(os.path.realpath(cwd / literal))
            if not resolved.is_relative_to(self._root):
                return False
            if not any(resolved.is_relative_to(allowed) for allowed in self._allowed):
                return False
        return True
