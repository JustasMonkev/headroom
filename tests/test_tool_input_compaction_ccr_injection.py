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
    assert merge_pipeline_ccr_hashes(["a", "b"], ["b", "c"]) == ["a", "b", "c"]
    assert merge_pipeline_ccr_hashes(None, None) == []
    assert merge_pipeline_ccr_hashes(["a"], [None, "", "a", "d"]) == ["a", "d"]
