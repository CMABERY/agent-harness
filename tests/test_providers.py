from __future__ import annotations

import json

import pytest

from agent.providers import (
    HttpRequest,
    HttpResponse,
    OpenAIChatProvider,
    OpenAIConfig,
    ProviderError,
    ScriptedProvider,
    TransportError,
    call,
    respond,
)
from agent.types import JSONObject, Message, TokenUsage, ToolCall, ToolSpec

TOOLS = (ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {}}),)
CONFIG = OpenAIConfig(base_url="https://llm.example/v1/", api_key="sk-test", model="m-1", max_retries=2)


def completion(message: JSONObject, finish: str = "tool_calls", usage: JSONObject | None = None) -> bytes:
    body: JSONObject = {"choices": [{"message": message, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body).encode()


class FakeTransport:
    def __init__(self, *replies: HttpResponse | TransportError) -> None:
        self.replies = list(replies)
        self.requests: list[HttpRequest] = []

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, TransportError):
            raise reply
        return reply


def provider(transport: FakeTransport, sleeps: list[float]) -> OpenAIChatProvider:
    return OpenAIChatProvider(CONFIG, transport=transport, sleep=sleeps.append)


# ------------------------------------------------------------- scripted


def test_scripted_provider_replays_records_and_assigns_ids() -> None:
    scripted = ScriptedProvider([respond(call("read_file", {"path": "a"}), call("read_file", {"path": "b"})),
                                 respond(text="hi")])
    first = scripted.complete([Message("user", "task")], TOOLS)
    assert [c.id for c in first.tool_calls] == ["call_1", "call_2"]
    assert json.loads(first.tool_calls[0].arguments) == {"path": "a"}
    assert scripted.complete([Message("user", "task")], TOOLS).text == "hi"
    assert len(scripted.requests) == 2 and scripted.requests[0].tools == TOOLS
    with pytest.raises(ProviderError, match="script exhausted"):
        scripted.complete([], TOOLS)


def test_scripted_provider_from_json() -> None:
    scripted = ScriptedProvider.from_json([
        {"text": "looking", "tool_calls": [{"name": "read_file", "arguments": {"path": "x"}}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"tool_calls": [{"name": "read_file", "arguments": "{not json"}]},
        {"error": "rate limited"},
    ])
    first = scripted.complete([], TOOLS)
    assert first.text == "looking" and first.usage == TokenUsage(10, 2)
    assert scripted.complete([], TOOLS).tool_calls[0].arguments == "{not json"
    with pytest.raises(ProviderError, match="rate limited"):
        scripted.complete([], TOOLS)
    with pytest.raises(ValueError):
        ScriptedProvider.from_json({"not": "a list"})


# ---------------------------------------------------------------- openai


def test_request_wire_format() -> None:
    transport = FakeTransport(HttpResponse(200, {}, completion({"content": "ok"}, "stop")))
    messages = [
        Message("system", "sys"),
        Message("user", "task"),
        Message("assistant", "", (ToolCall("c1", "read_file", '{"path": "a"}'),)),
        Message("tool", "result", tool_call_id="c1"),
    ]
    provider(transport, []).complete(messages, TOOLS)
    request = transport.requests[0]
    assert request.url == "https://llm.example/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(request.body)
    assert body["model"] == "m-1" and body["tool_choice"] == "auto"
    assert body["tools"] == [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                                               "parameters": {"type": "object", "properties": {}}}}]
    assert body["messages"][2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]}
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "result"}


def test_response_parsing_including_server_quirks() -> None:
    message: JSONObject = {
        "content": [{"type": "text", "text": "thinking"}],
        "tool_calls": [
            {"id": "a1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x"}'}},
            {"type": "function", "function": {"name": "read_file", "arguments": {"path": "y"}}},
            {"id": "a3", "type": "function", "function": {"name": "finish", "arguments": None}},
        ],
    }
    body = completion(message, usage={"prompt_tokens": 120, "completion_tokens": 30})
    result = provider(FakeTransport(HttpResponse(200, {}, body)), []).complete([], TOOLS)
    assert result.text == "thinking" and result.usage == TokenUsage(120, 30)
    assert [(c.id, c.arguments) for c in result.tool_calls] == [
        ("a1", '{"path": "x"}'), ("gen_1", '{"path": "y"}'), ("a3", "{}")]
    assert result.finish_reason == "tool_calls"


def test_retries_rate_limits_and_server_errors_then_succeeds() -> None:
    transport = FakeTransport(
        HttpResponse(429, {"Retry-After": "7"}, b"slow down"),
        TransportError("connection reset"),
        HttpResponse(200, {}, completion({"content": "ok"}, "stop")),
    )
    sleeps: list[float] = []
    assert provider(transport, sleeps).complete([], TOOLS).text == "ok"
    assert sleeps == [7.0, 2.0]


def test_gives_up_after_retries_with_last_error() -> None:
    transport = FakeTransport(*(HttpResponse(503, {}, b"overloaded") for _ in range(3)))
    with pytest.raises(ProviderError, match=r"no response after 3 attempt\(s\); last error: HTTP 503: overloaded"):
        provider(transport, []).complete([], TOOLS)


def test_client_errors_are_not_retried() -> None:
    transport = FakeTransport(HttpResponse(401, {}, b'{"error": "bad key"}'))
    with pytest.raises(ProviderError, match="HTTP 401"):
        provider(transport, []).complete([], TOOLS)
    assert len(transport.requests) == 1


def test_garbled_body_is_retried() -> None:
    transport = FakeTransport(HttpResponse(200, {}, b"<html>proxy"), HttpResponse(200, {}, completion({"content": "ok"})))
    assert provider(transport, []).complete([], TOOLS).text == "ok"


def test_config_from_environment() -> None:
    config = OpenAIConfig.from_env({"AGENT_BASE_URL": "http://localhost:8000/v1", "AGENT_MODEL": "local"})
    assert config.api_key == "" and config.model == "local"
    assert OpenAIConfig.from_env({"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m"}).base_url == "https://api.openai.com/v1"
    with pytest.raises(ProviderError, match="AGENT_MODEL"):
        OpenAIConfig.from_env({"AGENT_API_KEY": "k"})
    with pytest.raises(ProviderError, match="AGENT_API_KEY"):
        OpenAIConfig.from_env({"AGENT_MODEL": "m"})


def test_urllib_transport_against_a_loopback_server() -> None:
    """The real transport, over real HTTP on 127.0.0.1 (no external network)."""
    import http.server
    import threading

    replies = [(500, b"try again"), (200, completion({"content": "pong"}, "stop"))]
    seen: list[tuple[str, str, JSONObject]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server naming)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers["Authorization"], body))
            status, payload = replies.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = OpenAIConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="sk-local",
                              model="m", timeout_s=10, max_retries=1)
        result = OpenAIChatProvider(config, sleep=lambda _: None).complete([Message("user", "ping")], TOOLS)
    finally:
        server.shutdown()
        server.server_close()
    assert result.text == "pong"
    assert [path for path, _, _ in seen] == ["/v1/chat/completions"] * 2
    assert seen[0][1] == "Bearer sk-local" and seen[0][2]["messages"] == [{"role": "user", "content": "ping"}]


def test_retries_stop_at_the_callers_deadline() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    transport = FakeTransport(*(HttpResponse(503, {}, b"busy") for _ in range(5)))
    config = OpenAIConfig(base_url="https://llm.example/v1", api_key="k", model="m", timeout_s=120, max_retries=4)
    adapter = OpenAIChatProvider(config, transport=transport, sleep=sleep, now=lambda: now[0])
    with pytest.raises(ProviderError, match="no wall-clock budget left to retry"):
        adapter.complete([], TOOLS, timeout_s=5.0)
    assert sleeps == [1.0, 2.0]  # a third backoff (4 s) would end past the 5 s budget
    assert [r.timeout_s for r in transport.requests] == [5.0, 4.0, 2.0]  # each try fits the time left


def test_duplicate_or_missing_server_ids_are_replaced() -> None:
    message: JSONObject = {"content": None, "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
    ]}
    result = provider(FakeTransport(HttpResponse(200, {}, completion(message))), []).complete([], TOOLS)
    assert [c.id for c in result.tool_calls] == ["call_0", "gen_1", "gen_2"]
