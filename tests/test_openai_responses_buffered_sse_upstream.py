"""Issue #8: a ``stream: false`` /v1/responses request whose upstream answers
with a successful ``200 text/event-stream`` body must not become a 502.

The buffered path in ``_buffered_ccr_operation()`` used to call
``response.json()`` unconditionally and only caught
``(KeyError, TypeError, AttributeError)``, so ``json.JSONDecodeError`` escaped
to the outer handler and turned a perfectly good upstream reply into
``502 proxy_error``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.handlers.openai import _openai_responses_sse_to_response  # noqa: E402
from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

SSE_BODY = (
    "event: response.completed\n"
    'data: {"type":"response.completed","response":{"id":"resp_sse_repro",'
    '"output":[],"usage":{"input_tokens":2,"output_tokens":1}}}\n\n'
)


def _make_app():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    app.dependency_overrides[require_loopback] = lambda: None
    return app


def _patch_sse_upstream(app, body: str = SSE_BODY, status_code: int = 200):
    captured: dict = {}
    server = app.state.proxy

    async def fake_retry(method, url, headers, body_, stream=False, **kwargs):
        captured["body"] = body_
        return httpx.Response(
            status_code,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
            request=httpx.Request(method, url),
        )

    server._retry_request = fake_retry
    return captured


def test_buffered_stream_false_sse_upstream_is_not_a_502() -> None:
    app = _make_app()
    with TestClient(app) as client:
        _patch_sse_upstream(app)
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={"model": "gpt-5.4", "stream": False, "input": "hello"},
        )

    assert resp.status_code == 200, resp.text
    # The successful SSE body is preserved verbatim, content-type included.
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text == SSE_BODY
    assert "proxy_error" not in resp.text


def test_buffered_sse_upstream_usage_is_still_accounted() -> None:
    """Usage/telemetry comes from the terminal SSE event, not response.json()."""
    app = _make_app()
    recorded: list = []
    with TestClient(app) as client:
        _patch_sse_upstream(app)
        server = app.state.proxy
        original = server._record_request_outcome

        async def spy(outcome):
            recorded.append(outcome)
            return await original(outcome)

        server._record_request_outcome = spy
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={"model": "gpt-5.4", "stream": False, "input": "hello"},
        )

    assert resp.status_code == 200
    assert recorded, "no request outcome recorded"
    outcome = recorded[-1]
    assert outcome.optimized_tokens == 2
    assert outcome.output_tokens == 1


def test_buffered_json_upstream_still_parsed_normally() -> None:
    """The ordinary JSON buffered path is unchanged."""
    app = _make_app()
    payload = {
        "id": "resp_json",
        "object": "response",
        "output": [],
        "usage": {"input_tokens": 7, "output_tokens": 4},
    }
    server_app = app
    with TestClient(server_app) as client:
        server = app.state.proxy

        async def fake_retry(method, url, headers, body_, stream=False, **kwargs):
            return httpx.Response(200, json=payload, request=httpx.Request(method, url))

        server._retry_request = fake_retry
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={"model": "gpt-5.4", "stream": False, "input": "hello"},
        )

    assert resp.status_code == 200
    assert resp.json() == payload


def test_buffered_non_json_non_sse_upstream_does_not_500() -> None:
    """A 200 text/plain body is also not JSON; it must pass through, not 502."""
    app = _make_app()
    with TestClient(app) as client:
        server = app.state.proxy

        async def fake_retry(method, url, headers, body_, stream=False, **kwargs):
            return httpx.Response(
                200,
                content=b"not json at all",
                headers={"content-type": "text/plain"},
                request=httpx.Request(method, url),
            )

        server._retry_request = fake_retry
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={"model": "gpt-5.4", "stream": False, "input": "hello"},
        )

    assert resp.status_code == 200
    assert resp.text == "not json at all"


class TestSseToResponseHelper:
    def test_extracts_terminal_completed_response(self) -> None:
        out = _openai_responses_sse_to_response(SSE_BODY)
        assert out is not None
        assert out["id"] == "resp_sse_repro"
        assert out["usage"] == {"input_tokens": 2, "output_tokens": 1}

    def test_prefers_last_terminal_event_over_earlier_ones(self) -> None:
        body = (
            'event: response.created\ndata: {"type":"response.created",'
            '"response":{"id":"r","usage":{"input_tokens":0}}}\n\n'
            'event: response.completed\ndata: {"type":"response.completed",'
            '"response":{"id":"r","usage":{"input_tokens":9,"output_tokens":3}}}\n\n'
        )
        out = _openai_responses_sse_to_response(body)
        assert out is not None
        assert out["usage"]["input_tokens"] == 9

    def test_handles_missing_trailing_blank_line(self) -> None:
        body = (
            'event: response.completed\ndata: {"type":"response.completed",'
            '"response":{"id":"tail","usage":{"input_tokens":1}}}'
        )
        out = _openai_responses_sse_to_response(body)
        assert out is not None and out["id"] == "tail"

    def test_falls_back_to_incomplete_and_failed_events(self) -> None:
        body = (
            'event: response.incomplete\ndata: {"type":"response.incomplete",'
            '"response":{"id":"inc","usage":{"input_tokens":5}}}\n\n'
        )
        out = _openai_responses_sse_to_response(body)
        assert out is not None and out["id"] == "inc"

    def test_returns_none_for_garbage(self) -> None:
        assert _openai_responses_sse_to_response("not sse") is None
        assert _openai_responses_sse_to_response("") is None
        assert _openai_responses_sse_to_response("data: {oops\n\n") is None

    def test_accepts_bytes(self) -> None:
        out = _openai_responses_sse_to_response(SSE_BODY.encode())
        assert out is not None and out["id"] == "resp_sse_repro"

    def test_ignores_done_sentinel(self) -> None:
        body = SSE_BODY + "data: [DONE]\n\n"
        out = _openai_responses_sse_to_response(body)
        assert out is not None and out["id"] == "resp_sse_repro"

    def test_round_trips_with_responses_to_sse(self) -> None:
        from headroom.proxy.handlers.openai import _openai_responses_to_sse

        resp = {
            "id": "resp_rt",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        raw = b"".join(_openai_responses_to_sse(resp)).decode()
        out = _openai_responses_sse_to_response(raw)
        assert out is not None
        assert out["id"] == "resp_rt"
        assert out["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_buffered_stream_ccr_with_sse_upstream_passes_through() -> None:
    """stream:true + headroom_retrieve forces a buffered stream:false upstream
    call; if that upstream answers with SSE, forward it instead of 502-ing."""
    from headroom.ccr import CCR_TOOL_NAME

    app = _make_app()
    with TestClient(app) as client:
        captured = _patch_sse_upstream(app)
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={
                "model": "gpt-5.4",
                "stream": True,
                "input": "hello",
                "tools": [{"type": "function", "name": CCR_TOOL_NAME, "parameters": {}}],
            },
        )

    assert resp.status_code == 200, resp.text
    assert "proxy_error" not in resp.text
    assert "response.completed" in resp.text
    # The buffered path did force a non-streaming upstream request.
    if captured.get("body"):
        assert captured["body"].get("stream") is False


def test_delayed_buffered_sse_is_appended_after_keepalive() -> None:
    """A raw SSE result must survive when the keepalive already started."""
    from headroom.ccr import CCR_TOOL_NAME

    app = _make_app()
    server = app.state.proxy

    async def delayed_retry(method, url, headers, body_, stream=False, **kwargs):
        await asyncio.sleep(1.1)
        return httpx.Response(
            200,
            content=SSE_BODY.encode(),
            headers={"content-type": "text/event-stream"},
            request=httpx.Request(method, url),
        )

    server._retry_request = delayed_retry
    with TestClient(app) as client:
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={
                "model": "gpt-5.4",
                "stream": True,
                "input": "hello",
                "tools": [{"type": "function", "name": CCR_TOOL_NAME, "parameters": {}}],
            },
        )

    assert resp.status_code == 200
    assert '"type":"ping"' in resp.text
    assert "response.completed" in resp.text
    assert "proxy_error" not in resp.text


@pytest.mark.parametrize("client_stream", [True, False])
def test_sse_terminal_response_is_used_for_ccr_interception(client_stream: bool) -> None:
    """A retrieval call inside upstream SSE must be handled for either client mode."""
    from headroom.ccr import CCR_TOOL_NAME

    terminal = {
        "id": "resp_with_retrieve",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "name": CCR_TOOL_NAME,
                "call_id": "call_retrieve",
                "arguments": '{"hash":"abc"}',
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    upstream_sse = (
        "event: response.completed\n"
        f"data: {json.dumps({'type': 'response.completed', 'response': terminal})}\n\n"
    )
    final_response = {
        "id": "resp_after_retrieve",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": []}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }

    class _CCRSpy:
        config = type("C", (), {"enabled": True})()

        def __init__(self):
            self.handled = []

        def has_ccr_tool_calls(self, response, provider):  # noqa: ANN001
            return any(
                isinstance(item, dict) and item.get("name") == CCR_TOOL_NAME
                for item in response.get("output", [])
            )

        async def handle_response(self, response, *args, **kwargs):  # noqa: ANN001
            self.handled.append(response)
            return final_response

    app = _make_app()
    spy = _CCRSpy()
    app.state.proxy.ccr_response_handler = spy
    with TestClient(app) as client:
        _patch_sse_upstream(app, body=upstream_sse)
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={
                "model": "gpt-5.4",
                "stream": client_stream,
                "input": "hello",
                "tools": [{"type": "function", "name": CCR_TOOL_NAME, "parameters": {}}],
            },
        )

    assert resp.status_code == 200, resp.text
    assert spy.handled == [terminal]
    assert "resp_after_retrieve" in resp.text
    assert "call_retrieve" not in resp.text


def test_sse_body_is_not_reparsed_as_json_by_ccr() -> None:
    """CCR may inspect the terminal response without treating raw SSE as JSON."""
    app = _make_app()
    server = app.state.proxy
    calls: list = []

    class _Spy:
        config = type("C", (), {"enabled": True})()

        def has_ccr_tool_calls(self, resp, provider):  # noqa: ANN001
            calls.append(resp)
            return False

    server.ccr_response_handler = _Spy()
    with TestClient(app) as client:
        _patch_sse_upstream(app)
        resp = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={"model": "gpt-5.4", "stream": False, "input": "hello"},
        )

    assert resp.status_code == 200
    event = json.loads(SSE_BODY.split("data: ", 1)[1].strip())
    assert calls == [event["response"]]
    assert resp.text == SSE_BODY


def test_terminal_sse_memory_call_is_intercepted_and_continued() -> None:
    """Proxy-private memory calls in terminal SSE never leak to the client."""
    from types import SimpleNamespace

    initial = {
        "id": "resp_memory_call",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "name": "memory_search",
                "call_id": "call_memory",
                "arguments": '{"query":"preference"}',
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    final = {
        "id": "resp_after_memory",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "remembered"}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }

    def as_sse(payload: dict) -> bytes:
        event = {"type": "response.completed", "response": payload}
        return f"event: response.completed\ndata: {json.dumps(event)}\n\n".encode()

    class _MemorySpy:
        def __init__(self):
            self.config = SimpleNamespace(
                inject_context=False,
                inject_tools=True,
                project_root_override="",
            )
            self._backend = object()
            self.executed: list[tuple] = []

        def compute_memory_tool_definitions(self, provider):  # noqa: ANN001
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "memory_search",
                        "description": "Search memory.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        def has_memory_tool_calls(self, response, provider):  # noqa: ANN001
            return any(
                isinstance(item, dict) and item.get("name") == "memory_search"
                for item in response.get("output", [])
            )

        async def _ensure_initialized(self):
            return None

        async def ensure_initialized(self):
            return None

        def health_status(self):
            return {"initialized": False, "backend": "test"}

        async def close(self):
            return None

        async def _execute_memory_tool(self, name, args, user_id, provider):  # noqa: ANN001
            self.executed.append((name, args, user_id, provider))
            return json.dumps({"matches": ["remembered"]})

    app = _make_app()
    memory = _MemorySpy()
    app.state.proxy.memory_handler = memory
    calls: list[dict] = []

    async def fake_retry(method, url, headers, body_, stream=False, **kwargs):
        calls.append(body_)
        payload = initial if len(calls) == 1 else final
        return httpx.Response(
            200,
            content=as_sse(payload),
            headers={"content-type": "text/event-stream"},
            request=httpx.Request(method, url),
        )

    app.state.proxy._retry_request = fake_retry
    with TestClient(app) as client:
        resp = client.post(
            "/v1/responses",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
                "x-headroom-user-id": "user-1",
            },
            json={"model": "gpt-5.4", "stream": False, "store": True, "input": "hello"},
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["id"] == "resp_after_memory"
    assert len(calls) == 2
    assert memory.executed
    assert "memory_search" not in resp.text
