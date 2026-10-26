"""Model providers: the protocol, a scripted provider, and an OpenAI-compatible adapter.

Decisions owned here:
  * A provider either returns a ``ModelResponse`` or raises ``ProviderError``.
    Retries for transport failures and rate limits live in the adapter; the
    loop only sees "a response" or "no response is coming". ``timeout_s`` is
    the caller's remaining wall-clock budget, and the adapter's attempts and
    backoff sleeps must fit inside it.
  * The adapter does not judge the model's output. Arguments stay raw JSON
    text and are parsed by the loop, which owns the malformed-output policy.
    The adapter is lenient about *wire-format* variations between
    OpenAI-compatible servers (arguments sent as an object, missing or
    duplicate call ids, content as a list of parts), because those are server
    quirks the model cannot fix, not model errors.
"""

from __future__ import annotations

import http.client
import itertools
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from agent.types import JSONObject, JSONValue, Message, ModelResponse, TokenUsage, ToolCall, ToolSpec


class ProviderError(Exception):
    """No usable response could be obtained (network, auth, quota, exhausted script)."""


class LLMProvider(Protocol):
    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, timeout_s: float | None = None
    ) -> ModelResponse: ...


# --------------------------------------------------------------------------- #
# Scripted provider
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """A snapshot of exactly what the harness sent on one call."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]


type ScriptStep = ModelResponse | ProviderError | Callable[[ProviderRequest], ModelResponse]


class ScriptedProvider:
    """Replays a fixed sequence of responses and records every request.

    Tool calls with an empty id get a deterministic one (``call_1``, ...).
    Running past the end of the script raises ProviderError, so an
    under-scripted test fails with a clear outcome instead of hanging.
    """

    def __init__(self, steps: Iterable[ScriptStep]) -> None:
        self._steps = list(steps)
        self._ids = itertools.count(1)
        self.requests: list[ProviderRequest] = []

    @property
    def remaining(self) -> int:
        return len(self._steps)

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, timeout_s: float | None = None
    ) -> ModelResponse:
        request = ProviderRequest(tuple(messages), tuple(tools))
        self.requests.append(request)
        if not self._steps:
            raise ProviderError(f"script exhausted after {len(self.requests) - 1} responses")
        step = self._steps.pop(0)
        if isinstance(step, ProviderError):
            raise step
        response = step if isinstance(step, ModelResponse) else step(request)
        calls = tuple(c if c.id else replace(c, id=f"call_{next(self._ids)}") for c in response.tool_calls)
        return replace(response, tool_calls=calls)

    @classmethod
    def from_json(cls, data: JSONValue) -> ScriptedProvider:
        """Build from ``[{"text": ..., "tool_calls": [{"name", "arguments"}], "usage": {...}} | {"error": ...}]``."""
        if not isinstance(data, list):
            raise ValueError("a script must be a JSON array of steps")
        steps: list[ScriptStep] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"script step {index} must be an object")
            if "error" in item:
                steps.append(ProviderError(str(item["error"])))
                continue
            raw_calls = item.get("tool_calls", [])
            if not isinstance(raw_calls, list):
                raise ValueError(f"script step {index}: tool_calls must be an array")
            calls = []
            for raw in raw_calls:
                if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                    raise ValueError(f"script step {index}: each tool call needs a name")
                args = raw.get("arguments", {})
                calls.append(call(str(raw["name"]), args if isinstance(args, (dict, str)) else {},
                                  call_id=str(raw.get("id", ""))))
            usage = item.get("usage")
            steps.append(
                ModelResponse(
                    text=str(item.get("text", "")),
                    tool_calls=tuple(calls),
                    usage=_usage(usage) if isinstance(usage, dict) else None,
                    finish_reason=str(item.get("finish_reason", "tool_calls" if calls else "stop")),
                )
            )
        return cls(steps)


def call(name: str, arguments: JSONObject | str, *, call_id: str = "") -> ToolCall:
    """Build a tool call; a str ``arguments`` is passed through raw (for malformed-JSON tests)."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id, name=name, arguments=raw)


def respond(*calls: ToolCall, text: str = "", usage: TokenUsage | None = None) -> ModelResponse:
    return ModelResponse(text=text, tool_calls=calls, usage=usage,
                         finish_reason="tool_calls" if calls else "stop")


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat completions over urllib
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HttpRequest:
    url: str
    headers: Mapping[str, str]
    body: bytes
    timeout_s: float


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class TransportError(Exception):
    """The request did not produce an HTTP response (DNS, refused, reset, timeout)."""


type Transport = Callable[[HttpRequest], HttpResponse]


def urllib_transport(request: HttpRequest) -> HttpResponse:
    req = urllib.request.Request(request.url, data=request.body, headers=dict(request.headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=request.timeout_s) as resp:
            return HttpResponse(resp.status, dict(resp.headers.items()), resp.read())
    except urllib.error.HTTPError as err:
        headers = dict(err.headers.items()) if err.headers is not None else {}
        return HttpResponse(err.code, headers, err.read())
    except (OSError, http.client.HTTPException) as err:
        raise TransportError(f"{type(err).__name__}: {err}") from err


_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 120.0
    max_retries: int = 4

    @staticmethod
    def from_env(environ: Mapping[str, str]) -> OpenAIConfig:
        def pick(*names: str) -> str:
            return next((environ[n].strip() for n in names if environ.get(n, "").strip()), "")

        base_url = pick("AGENT_BASE_URL", "OPENAI_BASE_URL") or "https://api.openai.com/v1"
        api_key = pick("AGENT_API_KEY", "OPENAI_API_KEY")
        model = pick("AGENT_MODEL", "OPENAI_MODEL")
        if not model:
            raise ProviderError("set AGENT_MODEL (or OPENAI_MODEL) to the model name to use")
        host = urllib.parse.urlparse(base_url).hostname or ""
        if not api_key and host not in _LOCAL_HOSTS:
            raise ProviderError("set AGENT_API_KEY (or OPENAI_API_KEY); only localhost endpoints may omit it")
        return OpenAIConfig(base_url=base_url, api_key=api_key, model=model)


def _usage(raw: JSONObject) -> TokenUsage | None:
    prompt, completion = raw.get("prompt_tokens"), raw.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return TokenUsage(prompt, completion)
    return None


def _to_wire(message: Message) -> JSONObject:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    wire: JSONObject = {"role": message.role, "content": message.content}
    if message.role == "assistant" and message.tool_calls:
        wire["content"] = message.content or None
        wire["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in message.tool_calls
        ]
    return wire


class OpenAIChatProvider:
    def __init__(
        self,
        config: OpenAIConfig,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._transport = transport
        self._sleep = sleep
        self._now = now
        self._ids = itertools.count(1)

    def payload(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> JSONObject:
        body: JSONObject = {"model": self._config.model, "messages": [_to_wire(m) for m in messages]}
        if tools:
            body["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
            body["tool_choice"] = "auto"
        return body

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, timeout_s: float | None = None
    ) -> ModelResponse:
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        body = json.dumps(self.payload(messages, tools)).encode("utf-8")
        deadline = None if timeout_s is None else self._now() + timeout_s
        attempts = self._config.max_retries + 1
        last_error = "the wall-clock budget was already used up"
        tried = 0
        for attempt in range(attempts):
            remaining = None if deadline is None else deadline - self._now()
            if remaining is not None and remaining <= 0:
                break
            per_try = self._config.timeout_s if remaining is None else min(self._config.timeout_s, remaining)
            delay = min(2.0**attempt, 30.0)
            tried += 1
            try:
                response = self._transport(HttpRequest(url, headers, body, per_try))
            except TransportError as exc:
                last_error = f"network error: {exc}"
            else:
                excerpt = response.body[:500].decode("utf-8", errors="replace")
                if response.status == 200:
                    try:
                        return self.parse(response.body)
                    except ProviderError as exc:
                        last_error = str(exc)
                elif response.status in _RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status}: {excerpt}"
                    delay = _retry_after(response.headers, delay)
                else:
                    raise ProviderError(f"HTTP {response.status} from {url}: {excerpt}")
            if attempt + 1 < attempts:
                if deadline is not None and self._now() + delay >= deadline:
                    last_error += " (no wall-clock budget left to retry)"
                    break
                self._sleep(delay)
        raise ProviderError(f"no response after {tried} attempt(s); last error: {last_error}")

    def parse(self, body: bytes) -> ModelResponse:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderError(f"response body is not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ProviderError("response body is not a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            error = data.get("error")
            raise ProviderError(f"response has no choices{f': {error}' if error else ''}")
        choice = choices[0]
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        if isinstance(content, list):  # some servers send content parts
            text = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        else:
            text = content if isinstance(content, str) else ""
        calls: list[ToolCall] = []
        seen_ids: set[str] = set()
        raw_calls = message.get("tool_calls")
        for raw in raw_calls if isinstance(raw_calls, list) else []:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function")
            function = function if isinstance(function, dict) else {}
            arguments = function.get("arguments")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments)
            elif not isinstance(arguments, str) or not arguments.strip():
                arguments = "{}"
            call_id = raw.get("id")
            if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
                call_id = f"gen_{next(self._ids)}"  # the server's ids must be unique for our replies to be valid
            seen_ids.add(call_id)
            calls.append(ToolCall(id=call_id, name=str(function.get("name") or ""), arguments=arguments))
        usage = data.get("usage")
        finish = choice.get("finish_reason")
        return ModelResponse(
            text=text,
            tool_calls=tuple(calls),
            usage=_usage(usage) if isinstance(usage, dict) else None,
            finish_reason=finish if isinstance(finish, str) else "",
        )


def _retry_after(headers: Mapping[str, str], fallback: float) -> float:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return min(max(float(value), 0.0), 60.0)
            except ValueError:
                return fallback
    return fallback
