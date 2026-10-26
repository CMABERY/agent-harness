"""How untrusted content is presented to the model.

Decision: anything the harness did not author (file contents, file *names*,
command output, a subagent's prose) reaches the model only inside

    <untrusted-data source="file:src/app.py">
    ...
    </untrusted-data>

Two properties make the delimiter trustworthy:
  * every occurrence of the tag name inside the content is escaped
    (``<untrusted-data`` -> ``&lt;untrusted-data``, case-insensitively), so the
    content can neither close its block early nor open a fake trusted region;
  * the ``source`` attribute is HTML-escaped and stripped of control
    characters, because file names are attacker-controlled too.

Everything outside a block in a tool result is written by the harness. The
system prompt states this contract; ``tests/test_untrusted.py`` pins it.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

from agent.types import ToolResult, UntrustedText

TAG = "untrusted-data"
_TAG_PATTERN = re.compile(r"<(/?)(untrusted-data)", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_LINE_BREAKS = re.compile(r"[\r\n]")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def neutralize(text: str) -> str:
    return _TAG_PATTERN.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)


def _encodable(text: str, undecodable: str) -> str:
    """Replace lone surrogates so the text can be encoded as UTF-8 (JSON, logs, HTTP)."""
    try:
        return text.encode("utf-8", "surrogateescape").decode("utf-8", undecodable)
    except UnicodeEncodeError:  # surrogates that did not come from undecodable bytes
        return text.encode("utf-8", "replace").decode("utf-8")


def printable(text: str) -> str:
    """Make a value single-line and encodable: line breaks and control characters
    become visible escapes, and undecodable bytes in file names (which
    ``os.walk`` returns as lone surrogates) become ``\\xNN``."""
    text = _encodable(text, "backslashreplace")
    text = _LINE_BREAKS.sub(lambda m: "\\r" if m.group() == "\r" else "\\n", text)
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)


def inline(text: str) -> str:
    """For text the harness did not choose (a file name, a subagent's command)
    that must appear inside a trusted line: it can neither start a new line
    (and so forge a ``[harness note]``) nor open or close a data block."""
    return neutralize(printable(text))


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def wrap(item: UntrustedText) -> str:
    source = html.escape(printable(item.source), quote=True)
    body = neutralize(_encodable(item.text, "replace"))
    if body and not body.endswith("\n"):
        body += "\n"
    return f'<{TAG} source="{source}">\n{body}</{TAG}>'


def render_tool_result(name: str, result: ToolResult, notes: Sequence[str] = ()) -> str:
    """The exact text the model receives for one tool call."""
    command = result.effects.command
    if command is not None:  # a command ran: its exit status is the headline, not "ok"
        status = "timed out" if command.timed_out else f"exit {command.exit_code}"
    else:
        status = "ok" if result.ok else f"error {result.error_code or 'tool_error'}"
    parts = [f"[{name}: {status}] {result.message}"]
    parts.extend(wrap(item) for item in result.data)
    parts.extend(f"[harness note] {note}" for note in notes)
    return "\n".join(parts)
