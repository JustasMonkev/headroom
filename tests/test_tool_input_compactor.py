"""Tests for completed tool-call input compaction (F3).

Large historical tool-call arguments are replaced with CCR markers once
their matching results have arrived; pending calls, recent turns, and the
frozen cache prefix are never touched.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.config import ToolInputCompactionConfig
from headroom.transforms.tool_input_compactor import (
    CCR_INPUT_KEY,
    ToolInputCompactor,
)

LARGE_ARGS = json.dumps({"file_path": "/tmp/x.py", "content": "x" * 2000})
SMALL_ARGS = json.dumps({"file_path": "/tmp/x.py"})


class _FakeStore:
    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []

    def store(self, **kwargs: Any) -> str:
        self.stored.append(kwargs)
        return kwargs["explicit_hash"]


def _cfg(**overrides: Any) -> ToolInputCompactionConfig:
    defaults: dict[str, Any] = {"enabled": True, "min_chars": 100, "protect_recent_turns": 0}
    defaults.update(overrides)
    return ToolInputCompactionConfig(**defaults)


def _openai_conversation() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "write the file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Write", "arguments": LARGE_ARGS},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]


def _anthropic_conversation() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "write the file"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Write",
                    "input": {"file_path": "/tmp/x.py", "content": "y" * 2000},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
        {"role": "assistant", "content": "done"},
    ]


def test_openai_completed_call_is_compacted():
    store = _FakeStore()
    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=store).apply(messages)

    compacted = result.messages[1]["tool_calls"][0]
    assert compacted["id"] == "call_1"
    assert compacted["function"]["name"] == "Write"
    args = json.loads(compacted["function"]["arguments"])
    assert set(args) == {CCR_INPUT_KEY}
    assert "Retrieve original: hash=" in args[CCR_INPUT_KEY]
    assert result.compacted_count == 1
    assert result.transforms_applied == ["tool_input_compaction:Write"]
    assert len(result.ccr_hashes) == 1
    # Original bytes are retrievable from the store.
    assert store.stored[0]["original"] == LARGE_ARGS
    assert store.stored[0]["tool_call_id"] == "call_1"
    assert store.stored[0]["compression_strategy"] == "tool_input_compaction"
    # Input list is not mutated in place.
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == LARGE_ARGS


def test_anthropic_completed_call_is_compacted():
    store = _FakeStore()
    messages = _anthropic_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=store).apply(messages)

    block = result.messages[1]["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_1"
    assert block["name"] == "Write"
    assert set(block["input"]) == {CCR_INPUT_KEY}
    assert "Retrieve original: hash=" in block["input"][CCR_INPUT_KEY]
    assert result.compacted_count == 1
    # Original serialized input is retrievable.
    assert json.loads(store.stored[0]["original"])["file_path"] == "/tmp/x.py"


def test_pending_call_is_never_compacted():
    # No tool result for the call: arguments are live working context.
    messages = _openai_conversation()
    del messages[2]  # remove the tool result
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0
    assert result.messages is messages


def test_small_arguments_are_left_alone():
    messages = _openai_conversation()
    messages[1]["tool_calls"][0]["function"]["arguments"] = SMALL_ARGS
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0


def test_protect_recent_turns_skips_trailing_assistant_messages():
    messages = _openai_conversation()
    # Both assistant messages fall inside the protection window of 2.
    result = ToolInputCompactor(_cfg(protect_recent_turns=2), compression_store=_FakeStore()).apply(
        messages
    )
    assert result.compacted_count == 0


def test_frozen_prefix_is_never_mutated():
    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        messages, frozen_message_count=2
    )
    assert result.compacted_count == 0


def test_disabled_is_noop():
    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(enabled=False), compression_store=_FakeStore()).apply(messages)
    assert result.messages is messages
    assert result.compacted_count == 0


def test_idempotent_on_already_compacted_input():
    store = _FakeStore()
    compactor = ToolInputCompactor(_cfg(), compression_store=store)
    first = compactor.apply(_openai_conversation())
    second = compactor.apply(first.messages)
    assert second.compacted_count == 0
    assert len(store.stored) == 1

    anthropic_first = compactor.apply(_anthropic_conversation())
    anthropic_second = compactor.apply(anthropic_first.messages)
    assert anthropic_second.compacted_count == 0


def test_store_failure_does_not_break_compaction():
    class _BrokenStore:
        def store(self, **kwargs: Any) -> str:
            raise RuntimeError("boom")

    result = ToolInputCompactor(_cfg(), compression_store=_BrokenStore()).apply(
        _openai_conversation()
    )
    # Marker still emitted with the locally computed hash.
    assert result.compacted_count == 1
    args = json.loads(result.messages[1]["tool_calls"][0]["function"]["arguments"])
    assert "Retrieve original: hash=" in args[CCR_INPUT_KEY]


def test_result_before_call_does_not_count_as_completed():
    # A stray result at an EARLIER index must not mark the call completed.
    messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": "stale"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Write", "arguments": LARGE_ARGS},
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0
