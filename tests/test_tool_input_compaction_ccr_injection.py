"""Codex P1: the retrieve tool must be injected for tool-INPUT markers too.

`CCRToolInjector.scan_for_markers` reads message text and tool-RESULT content.
The tool-input compaction pass writes its marker inside
`tool_use.input` / `tool_calls[].function.arguments`, which the scanner never
visits — so on the first compaction of a session `detected_hashes` was empty,
both provider handlers skipped the sticky `headroom_retrieve` injection, and the
stored original was unreachable. The handlers now merge
hashes found in the final forwarded tool arguments into the injection decision.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.ccr.tool_injection import CCR_SYSTEM_INSTRUCTIONS, CCR_TOOL_NAME, CCRToolInjector
from headroom.proxy.ccr_marker_policy import has_new_ccr_markers, should_inject_ccr_tool
from headroom.proxy.helpers import (
    _reset_session_ccr_tracker_for_test,
    apply_session_sticky_ccr_tool,
)
from headroom.transforms.tool_input_compactor import (
    CCR_INPUT_KEY,
    ccr_hashes_from_markers,
    ccr_hashes_in_tool_arguments,
    merge_pipeline_ccr_hashes,
)

HASH = "abc123def456abc123def456"
MARKER = f"[tool input elided. Retrieve original: hash={HASH}]"


@pytest.fixture(autouse=True)
def _fresh_tracker() -> Any:
    _reset_session_ccr_tracker_for_test()
    yield
    _reset_session_ccr_tracker_for_test()


def _anthropic_messages() -> list[dict[str, Any]]:
    """Conversation whose ONLY CCR marker lives inside a tool_use input."""
    return [
        {"role": "user", "content": "search the repo"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Grep",
                    "input": {CCR_INPUT_KEY: MARKER},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
    ]


def _openai_messages() -> list[dict[str, Any]]:
    import json

    return [
        {"role": "user", "content": "search the repo"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Grep",
                        "arguments": json.dumps({CCR_INPUT_KEY: MARKER}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]


@pytest.mark.parametrize(
    ("provider", "messages"),
    [("anthropic", _anthropic_messages()), ("openai", _openai_messages())],
)
def test_scanner_alone_misses_tool_input_markers(provider: str, messages: Any) -> None:
    """Documents the gap the handler fix compensates for."""
    injector = CCRToolInjector(
        provider=provider, inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers(messages)
    assert injector.detected_hashes == []
    assert injector.has_compressed_content is False


@pytest.mark.parametrize(
    ("provider", "messages"),
    [("anthropic", _anthropic_messages()), ("openai", _openai_messages())],
)
def test_retrieve_tool_is_injected_for_a_tool_input_only_marker(
    provider: str, messages: Any
) -> None:
    injector = CCRToolInjector(
        provider=provider, inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers(messages)

    # The handler merges hashes found in the final forwarded tool arguments.
    detected = merge_pipeline_ccr_hashes(
        injector.detected_hashes,
        ccr_hashes_in_tool_arguments(messages),
    )
    assert detected == [HASH]

    has_new = has_new_ccr_markers(
        current_detected_hashes=detected,
        previous_forwarded_messages=None,
        provider=provider,  # type: ignore[arg-type]
    )
    assert has_new is True

    # Even with a frozen prefix (injection normally deferred), a fresh marker
    # must override the deferral — otherwise the marker is unredeemable.
    should_inject, is_override = should_inject_ccr_tool(
        configured_inject_tool=True,
        frozen_message_count=2,
        has_compressed_content=has_new,
    )
    assert should_inject is True
    assert is_override is True

    tools, injected = apply_session_sticky_ccr_tool(
        provider=provider,  # type: ignore[arg-type]
        session_id="sess-1",
        request_id="req-1",
        existing_tools=[],
        has_compressed_content_this_turn=has_new,
    )
    assert injected is True
    names = {t.get("name") or (t.get("function") or {}).get("name") for t in tools}
    assert CCR_TOOL_NAME in names


def test_without_the_merge_the_tool_is_not_injected() -> None:
    """The pre-fix behaviour, pinned so the regression can't silently return."""
    injector = CCRToolInjector(
        provider="anthropic", inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers(_anthropic_messages())
    has_new = has_new_ccr_markers(
        current_detected_hashes=injector.detected_hashes,
        previous_forwarded_messages=None,
        provider="anthropic",
    )
    assert has_new is False
    should_inject, _ = should_inject_ccr_tool(
        configured_inject_tool=True,
        frozen_message_count=2,
        has_compressed_content=has_new,
    )
    assert should_inject is False  # <- the bug


def test_merge_is_order_stable_and_deduplicated() -> None:
    a, b, c, d = (
        "aaaaaaaaaaaa",
        "bbbbbbbbbbbb",
        "cccccccccccc",
        "dddddddddddd",
    )
    assert merge_pipeline_ccr_hashes([a, b], [b, c]) == [a, b, c]
    assert merge_pipeline_ccr_hashes(None, None) == []
    assert merge_pipeline_ccr_hashes([a], [None, "", a, d]) == [a, d]


# ---------------------------------------------------------------------------
# #1850 regression: only REDEEMABLE hashes may drive the injection decision.
#
# `markers_inserted` is a mixed bag. SmartCrusher appends
# `<headroom:tool_digest sha256="…">` provenance strings that
# `CCRToolInjector.scan_for_markers` can never return, so an unfiltered merge
# made every such entry unconditionally "new" — re-injecting `headroom_retrieve`
# on a byte-identical replayed prefix and busting the tools cache segment the
# frozen prefix exists to protect.
# ---------------------------------------------------------------------------


def test_non_hash_markers_are_not_treated_as_ccr_hashes() -> None:
    from headroom.utils import create_tool_digest_marker

    digest = create_tool_digest_marker("0123456789ab")
    assert ccr_hashes_from_markers([digest]) == []
    assert ccr_hashes_from_markers(["stable_prefix_hash:0123456789abcdef"]) == []
    assert ccr_hashes_from_markers([HASH, digest, None, 7, HASH]) == [HASH]
    # SmartCrusher's 12-hex short hashes and the 24-hex form both survive.
    assert ccr_hashes_from_markers(["0123456789ab"]) == ["0123456789ab"]
    # The scanner's generic bracket pattern is IGNORECASE, so upper-case hex
    # hashes are reachable and must not be filtered out.
    assert ccr_hashes_from_markers(["ABC123DEF456ABC123DEF456"]) == ["ABC123DEF456ABC123DEF456"]


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_replayed_identical_prefix_does_not_reinject_the_retrieve_tool(provider: str) -> None:
    """A byte-identical frozen prefix must NOT look like it carries new markers.

    SmartCrusher crushed something earlier in the session, so `markers_inserted`
    carries a `tool_digest` provenance string every turn. The forwarded bytes are
    unchanged, so nothing new is redeemable and the tools cache segment must
    stay intact.
    """
    from headroom.utils import create_tool_digest_marker

    messages = _anthropic_messages() if provider == "anthropic" else _openai_messages()
    # What the pipeline reports this turn: provenance only, no new CCR hash.
    markers_inserted = [create_tool_digest_marker("0123456789ab"), "stable_prefix_hash:deadbeef"]

    injector = CCRToolInjector(
        provider=provider, inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers(messages)
    detected = merge_pipeline_ccr_hashes(injector.detected_hashes, markers_inserted)
    assert detected == []

    has_new = has_new_ccr_markers(
        current_detected_hashes=detected,
        previous_forwarded_messages=messages,
        provider=provider,  # type: ignore[arg-type]
    )
    assert has_new is False

    should_inject, is_override = should_inject_ccr_tool(
        configured_inject_tool=True,
        frozen_message_count=len(messages),
        has_compressed_content=has_new,
    )
    assert should_inject is False
    assert is_override is False


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_a_genuine_tool_input_marker_still_reinjects_on_a_replayed_prefix(provider: str) -> None:
    """The point of the original fix: a real minted hash still forces injection.

    Same replayed prefix as above, but this turn the tool-input compactor minted
    a real CCR hash that lives only inside `tool_use.input` — invisible to the
    scanner on BOTH the current and the previous messages. It must still count
    as new, or the agent holds a marker it cannot redeem.
    """
    from headroom.utils import create_tool_digest_marker

    messages = _anthropic_messages() if provider == "anthropic" else _openai_messages()
    fresh = "f00dcafef00dcafef00dcafe"
    markers_inserted = [create_tool_digest_marker("0123456789ab"), fresh]

    injector = CCRToolInjector(
        provider=provider, inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers(messages)
    detected = merge_pipeline_ccr_hashes(injector.detected_hashes, markers_inserted)
    assert detected == [fresh]

    has_new = has_new_ccr_markers(
        current_detected_hashes=detected,
        previous_forwarded_messages=messages,
        provider=provider,  # type: ignore[arg-type]
    )
    assert has_new is True

    should_inject, is_override = should_inject_ccr_tool(
        configured_inject_tool=True,
        frozen_message_count=len(messages),
        has_compressed_content=has_new,
    )
    assert should_inject is True
    assert is_override is True


# ---------------------------------------------------------------------------
# Codex P2: a REPLAYED compacted tool input must still register.
#
# `merge_pipeline_ccr_hashes` only covers hashes minted during the current
# pipeline run. When a conversation that already contains a compacted
# `tool_calls[].function.arguments` / `tool_use.input` reaches a new worker or a
# restarted process, the compactor skips the existing `_ccr` value as idempotent
# (nothing minted), the in-memory session tracker is empty (no sticky replay),
# and `scan_for_markers` cannot see the marker — so the retrieval tool was never
# injected even though the marker and its persistent CCR entry are still there.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "messages"),
    [("anthropic", _anthropic_messages()), ("openai", _openai_messages())],
)
def test_existing_tool_arguments_are_scanned_for_ccr_hashes(provider: str, messages: Any) -> None:
    assert ccr_hashes_in_tool_arguments(messages) == [HASH]


def test_tool_argument_scan_ignores_non_ccr_content() -> None:
    import json

    from headroom.utils import create_tool_digest_marker

    junk = [
        {"role": "user", "content": f"hash={HASH}"},  # text is the scanner's job
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {
                        "name": "Grep",
                        "arguments": json.dumps(
                            {
                                "pattern": create_tool_digest_marker("0123456789ab"),
                                "note": "stable_prefix_hash:deadbeef",
                                "short": "Retrieve original: hash=abc",  # too short
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": f"<<ccr:{HASH}>>"}]},
        "not-a-message",
        {"role": "assistant", "tool_calls": [None, {"function": {"arguments": None}}]},
    ]
    assert ccr_hashes_in_tool_arguments(junk) == []


def test_tool_argument_scan_reads_both_marker_spellings() -> None:
    import json

    other = "0123456789ab"
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Grep", "arguments": json.dumps({CCR_INPUT_KEY: MARKER})},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "Grep",
                        "arguments": f'{{"{CCR_INPUT_KEY}":"<<ccr:{other},x,1>>"}}',
                    },
                },
            ],
        },
        # De-duplicated across shapes and messages.
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Grep", "input": {CCR_INPUT_KEY: MARKER}}
            ],
        },
    ]
    assert ccr_hashes_in_tool_arguments(messages) == [HASH, other]


# ---------------------------------------------------------------------------
# Codex P2: only the `_ccr` PROPERTY counts, not the whole arguments blob.
#
# Marker text is ordinary data for an ordinary tool call. This repo's own
# agents grep for `<<ccr:` and `Retrieve original: hash=` while auditing the
# markers, so `Grep(pattern="<<ccr:0123456789ab>>")` used to hand back a hash
# lifted straight out of its own search pattern. Nothing was ever stored under
# it, but `apply_session_sticky_ccr_tool` registers it, so the useless
# `headroom_retrieve` tool then rides along for the rest of the session and
# counts toward the Anthropic tool-search threshold.
# ---------------------------------------------------------------------------

PHANTOM = "0123456789ab"


def _marker_hunting_arguments() -> dict[str, Any]:
    """A completed, uncompacted tool call that merely TALKS about markers."""
    return {
        "pattern": f"<<ccr:{PHANTOM}>>",
        "path": "headroom/transforms/tool_input_compactor.py",
        "note": f"Retrieve original: hash={HASH} is the other spelling",
    }


def test_marker_text_outside_the_ccr_property_yields_no_hashes_openai() -> None:
    import json

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_grep",
                    "type": "function",
                    "function": {
                        "name": "Grep",
                        "arguments": json.dumps(_marker_hunting_arguments()),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_grep", "content": "3 matches"},
    ]
    assert ccr_hashes_in_tool_arguments(messages) == []


def test_marker_text_outside_the_ccr_property_yields_no_hashes_anthropic() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_grep",
                    "name": "Grep",
                    "input": _marker_hunting_arguments(),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_grep", "content": "ok"}],
        },
    ]
    assert ccr_hashes_in_tool_arguments(messages) == []


@pytest.mark.parametrize(
    ("provider", "messages"),
    [("anthropic", _anthropic_messages()), ("openai", _openai_messages())],
)
def test_a_real_ccr_property_still_yields_its_hash(provider: str, messages: Any) -> None:
    """The behaviour the function exists for, kept while the blob scan goes.

    A genuinely compacted argument reaching a fresh worker must still surface
    its hash so `headroom_retrieve` is injected and the marker stays redeemable.
    """
    assert ccr_hashes_in_tool_arguments(messages) == [HASH]


def test_a_real_marker_survives_alongside_a_marker_hunting_call() -> None:
    """Both directions in one transcript: the phantom drops, the real one stays."""
    import json

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_grep",
                    "type": "function",
                    "function": {
                        "name": "Grep",
                        "arguments": json.dumps(_marker_hunting_arguments()),
                    },
                },
                {
                    "id": "call_compacted",
                    "type": "function",
                    "function": {"name": "Read", "arguments": json.dumps({CCR_INPUT_KEY: MARKER})},
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_grep",
                    "name": "Grep",
                    "input": _marker_hunting_arguments(),
                },
                {
                    "type": "tool_use",
                    "id": "toolu_compacted",
                    "name": "Read",
                    "input": {CCR_INPUT_KEY: MARKER},
                },
            ],
        },
    ]
    assert ccr_hashes_in_tool_arguments(messages) == [HASH]


def test_malformed_arguments_do_not_raise() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                # `_ccr` present so the parse gate opens, but the blob is junk.
                {"function": {"name": "x", "arguments": '{"_ccr": '}},
                {"function": {"name": "x", "arguments": f"not json at all _ccr {MARKER}"}},
                {"function": {"name": "x", "arguments": '["_ccr", "' + MARKER + '"]'}},
                {"function": {"name": "x", "arguments": '{"_ccr": {"nested": "' + MARKER + '"}}'}},
                {"function": {"name": "x", "arguments": None}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t", "name": "x", "input": None},
                {"type": "tool_use", "id": "t", "name": "x", "input": f"raw {MARKER}"},
                {"type": "tool_use", "id": "t", "name": "x", "input": {CCR_INPUT_KEY: 7}},
            ],
        },
    ]
    assert ccr_hashes_in_tool_arguments(messages) == []


def test_anthropic_input_as_a_json_string_still_reads_the_ccr_property() -> None:
    """Streaming accumulators can hand `input` back as raw partial JSON."""
    import json

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": json.dumps({CCR_INPUT_KEY: MARKER}),
                }
            ],
        }
    ]
    assert ccr_hashes_in_tool_arguments(messages) == [HASH]


@pytest.mark.parametrize(
    ("provider", "messages"),
    [("anthropic", _anthropic_messages()), ("openai", _openai_messages())],
)
def test_replayed_compacted_input_injects_the_tool_after_a_restart(
    provider: str, messages: Any
) -> None:
    """The restart case end-to-end, at the level the handlers compose it.

    Nothing is minted this turn (`pipeline_ccr_hashes` empty) and the tracker is
    fresh, so before the fix `detected` was empty and the marker in the
    transcript stayed unredeemable.
    """
    injector = CCRToolInjector(
        provider=provider, inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers(messages)
    assert injector.detected_hashes == []

    detected = merge_pipeline_ccr_hashes(
        injector.detected_hashes,
        [*[], *ccr_hashes_in_tool_arguments(messages)],
    )
    assert detected == [HASH]

    has_new = has_new_ccr_markers(
        current_detected_hashes=detected,
        previous_forwarded_messages=None,
        provider=provider,  # type: ignore[arg-type]
    )
    assert has_new is True

    tools, injected = apply_session_sticky_ccr_tool(
        provider=provider,  # type: ignore[arg-type]
        session_id="sess-restart",
        request_id="req-1",
        existing_tools=[],
        has_compressed_content_this_turn=has_new,
    )
    assert injected is True
    names = {t.get("name") or (t.get("function") or {}).get("name") for t in tools}
    assert CCR_TOOL_NAME in names


def test_openai_handler_injects_retrieve_tool_for_a_replayed_compacted_input() -> None:
    """Wiring check: the OpenAI chat path itself must scan tool arguments.

    Simulates the restart: the pipeline reports nothing minted (the compactor
    skips the existing `_ccr` value as idempotent) and the process-wide session
    CCR tracker is fresh, so the ONLY evidence that CCR is live in this
    conversation is the marker inside `tool_calls[].function.arguments`.
    """
    pytest.importorskip("fastapi")

    from types import SimpleNamespace

    import httpx
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=True,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    forwarded: dict[str, Any] = {}

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy

        def _fake_apply(**kwargs: Any) -> Any:
            # Nothing compacted this turn, nothing minted: the marker is
            # historical, replayed from the client's transcript.
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=100,
                tokens_after=100,
                waste_signals=None,
                markers_inserted=[],
            )

        proxy.openai_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            forwarded["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 3, "total_tokens": 103},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [*_openai_messages(), {"role": "user", "content": "and now?"}],
            },
        )
        assert response.status_code == 200

    tools = forwarded["body"].get("tools") or []
    names = {t.get("name") or (t.get("function") or {}).get("name") for t in tools}
    assert CCR_TOOL_NAME in names, forwarded["body"].get("tools")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_handler_rollback_uses_only_hashes_in_forwarded_messages(provider: str) -> None:
    """Rollback neither makes stale hashes sticky nor hides original markers."""
    pytest.importorskip("fastapi")

    import json
    from types import SimpleNamespace

    import httpx
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=True,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=True,
        ccr_inject_system_instructions=True,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    forwarded: list[dict[str, Any]] = []
    pipeline_calls = 0
    original_messages = [{"role": "user", "content": "hello"}]
    inflated_messages = [{"role": "user", "content": f"{MARKER}\n" + ("inflated " * 10_000)}]

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy

        def _fake_apply(**kwargs: Any) -> Any:
            nonlocal pipeline_calls
            pipeline_calls += 1
            if pipeline_calls == 2:
                return SimpleNamespace(
                    messages=kwargs["messages"],
                    transforms_applied=[],
                    timing={},
                    tokens_before=10,
                    tokens_after=10,
                    waste_signals=None,
                    markers_inserted=[],
                )
            return SimpleNamespace(
                messages=inflated_messages,
                transforms_applied=["fake:ccr"],
                timing={},
                tokens_before=1,
                tokens_after=1_000_000,
                waste_signals=None,
                markers_inserted=[HASH],
            )

        if provider == "openai":
            proxy.openai_pipeline.apply = _fake_apply
        else:
            proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            forwarded.append(json.loads(json.dumps(body)))
            if provider == "openai":
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl_rollback",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                            "total_tokens": 13,
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "msg_rollback",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry
        path = "/v1/chat/completions" if provider == "openai" else "/v1/messages"
        headers = (
            {"authorization": "Bearer test-key"}
            if provider == "openai"
            else {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
        )
        headers["x-headroom-session-id"] = "rollback-session"

        def _post(messages: list[dict[str, Any]]) -> None:
            payload: dict[str, Any] = {
                "model": "gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-6",
                "messages": messages,
            }
            if provider == "anthropic":
                payload["max_tokens"] = 64
            response = client.post(path, headers=headers, json=payload)
            assert response.status_code == 200, response.text

        _post(original_messages)
        _post([{"role": "user", "content": "still no marker"}])
        _post(_openai_messages() if provider == "openai" else _anthropic_messages())

    assert len(forwarded) == 3
    for body in forwarded[:2]:
        assert HASH not in json.dumps(body["messages"])
        assert CCR_SYSTEM_INSTRUCTIONS not in json.dumps(body["messages"])
        assert CCR_SYSTEM_INSTRUCTIONS not in json.dumps(body.get("system"))
        names = {
            t.get("name") or (t.get("function") or {}).get("name") for t in body.get("tools") or []
        }
        assert CCR_TOOL_NAME not in names

    marker_body = forwarded[2]
    assert HASH in json.dumps(marker_body["messages"])
    if provider == "anthropic":
        assert {message["role"] for message in marker_body["messages"]} <= {
            "user",
            "assistant",
        }
        assert CCR_SYSTEM_INSTRUCTIONS not in json.dumps(marker_body["messages"])
        assert CCR_SYSTEM_INSTRUCTIONS in json.dumps(marker_body["system"])
    else:
        # OpenAI Chat keeps its system guidance in `messages`; this change is
        # Anthropic-only.
        assert CCR_SYSTEM_INSTRUCTIONS in json.dumps(marker_body["messages"])
    marker_names = {
        t.get("name") or (t.get("function") or {}).get("name")
        for t in marker_body.get("tools") or []
    }
    assert CCR_TOOL_NAME in marker_names


@pytest.mark.parametrize(
    ("initial_system", "expected_system"),
    [
        (None, CCR_SYSTEM_INSTRUCTIONS),
        ("base system", f"base system\n\n{CCR_SYSTEM_INSTRUCTIONS}"),
        (
            [
                {
                    "type": "text",
                    "text": "base system",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            [
                {
                    "type": "text",
                    "text": "base system",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": CCR_SYSTEM_INSTRUCTIONS},
            ],
        ),
    ],
)
def test_anthropic_ccr_guidance_is_final_and_cache_stable(
    initial_system: Any,
    expected_system: Any,
) -> None:
    """Absent/string/list systems stay valid and byte-stable on a frozen replay."""
    pytest.importorskip("fastapi")

    import copy

    import httpx
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    class _FrozenTracker:
        def __init__(self) -> None:
            self._cached_token_count = 0
            self.forwarded: list[dict[str, Any]] = []

        def get_frozen_message_count(self) -> int:
            return 1

        def get_last_original_messages(self) -> list[dict[str, Any]]:
            return []

        def get_last_forwarded_messages(self) -> list[dict[str, Any]]:
            return self.forwarded.copy()

        def update_from_response(self, **kwargs: Any) -> None:
            self._cached_token_count = kwargs.get("cache_read_tokens", 0) + kwargs.get(
                "cache_write_tokens", 0
            )
            self.forwarded = kwargs.get("messages", []).copy()

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=True,
        ccr_inject_system_instructions=True,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    forwarded: list[dict[str, Any]] = []

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        tracker = _FrozenTracker()
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "stable-ccr-system"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            forwarded.append(copy.deepcopy(body))
            return httpx.Response(
                200,
                json={
                    "id": "msg_ccr_system",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry
        payload: dict[str, Any] = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": _anthropic_messages(),
        }
        if initial_system is not None:
            payload["system"] = initial_system

        for _ in range(2):
            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json=payload,
            )
            assert response.status_code == 200, response.text

    assert [body["system"] for body in forwarded] == [expected_system, expected_system]
    for body in forwarded:
        assert {message["role"] for message in body["messages"]} <= {"user", "assistant"}
        assert str(body["system"]).count(CCR_SYSTEM_INSTRUCTIONS) == 1
    assert CCR_TOOL_NAME in {tool.get("name") for tool in forwarded[0].get("tools") or []}


def test_anthropic_system_compaction_marker_gets_final_guidance_and_tool(monkeypatch) -> None:
    """A marker minted by Layer 3 is redeemable in the same forwarded request."""
    pytest.importorskip("fastapi")

    import copy

    import httpx
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
    monkeypatch.setattr(
        "headroom.transforms.compression_units.find_content_router",
        lambda pipeline: object(),
    )

    def _compact(body: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], bool, int, int]:
        compacted = {**body, "system": f"compacted system\n\n{MARKER}"}
        return compacted, True, 100, 20

    monkeypatch.setattr("headroom.proxy.system_compaction.compact_system_prompt", _compact)

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=True,
        ccr_inject_system_instructions=True,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    forwarded: dict[str, Any] = {}

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            forwarded.update(copy.deepcopy(body))
            return httpx.Response(
                200,
                json={
                    "id": "msg_system_compaction",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "system": "base system",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text

    assert MARKER in forwarded["system"]
    assert forwarded["system"].endswith(CCR_SYSTEM_INSTRUCTIONS)
    assert {message["role"] for message in forwarded["messages"]} <= {"user", "assistant"}
    assert CCR_TOOL_NAME in {tool.get("name") for tool in forwarded.get("tools") or []}
