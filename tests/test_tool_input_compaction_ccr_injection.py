"""Codex P1: the retrieve tool must be injected for tool-INPUT markers too.

`CCRToolInjector.scan_for_markers` reads message text and tool-RESULT content.
The tool-input compaction pass writes its marker inside
`tool_use.input` / `tool_calls[].function.arguments`, which the scanner never
visits — so on the first compaction of a session `detected_hashes` was empty,
both provider handlers skipped the sticky `headroom_retrieve` injection, and the
stored original was unreachable. The handlers now merge
`TransformResult.markers_inserted` into the injection decision.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.ccr.tool_injection import CCR_TOOL_NAME, CCRToolInjector
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

    # The handler merges the hashes the pipeline reported minting.
    detected = merge_pipeline_ccr_hashes(injector.detected_hashes, [HASH])
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
