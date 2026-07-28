"""Issue #8: a ``stream: false`` /v1/responses request whose upstream answers
with a successful ``200 text/event-stream`` body must not become a 502.

The buffered path in ``_buffered_ccr_operation()`` used to call
``response.json()`` unconditionally and only caught
``(KeyError, TypeError, AttributeError)``, so ``json.JSONDecodeError`` escaped
to the outer handler and turned a perfectly good upstream reply into
``502 proxy_error``.
"""

from __future__ import annotations

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
            return httpx.Response(
                200, json=payload, request=httpx.Request(method, url)
            )

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


def test_sse_body_is_not_reparsed_as_json_by_ccr(monkeypatch) -> None:
    """An SSE upstream must not be fed into the CCR/memory JSON rewrite paths."""
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
    # resp_json stays None for an SSE upstream, so the CCR probe never runs.
    assert calls == []
    assert json.loads(SSE_BODY.split("data: ", 1)[1].strip())["type"] == "response.completed"
