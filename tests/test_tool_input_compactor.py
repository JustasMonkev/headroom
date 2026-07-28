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

# Read-only (reproducible) arguments — the only kind that may be compacted.
LARGE_ARGS = json.dumps({"pattern": "def handler", "path": "/repo", "glob": "y" * 2000})
SMALL_ARGS = json.dumps({"pattern": "def handler"})
# A mutating call: its arguments are the sole exact record of the change.
MUTATING_ARGS = json.dumps({"file_path": "/tmp/x.py", "content": "x" * 2000})


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
                    "function": {"name": "Grep", "arguments": LARGE_ARGS},
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
                    "name": "Grep",
                    "input": {"pattern": "def handler", "path": "/repo", "glob": "y" * 2000},
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
    assert compacted["function"]["name"] == "Grep"
    args = json.loads(compacted["function"]["arguments"])
    assert set(args) == {CCR_INPUT_KEY}
    assert "Retrieve original: hash=" in args[CCR_INPUT_KEY]
    assert result.compacted_count == 1
    assert result.transforms_applied == ["tool_input_compaction:Grep"]
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
    assert block["name"] == "Grep"
    assert set(block["input"]) == {CCR_INPUT_KEY}
    assert "Retrieve original: hash=" in block["input"][CCR_INPUT_KEY]
    assert result.compacted_count == 1
    # Original serialized input is retrievable.
    assert json.loads(store.stored[0]["original"])["pattern"] == "def handler"


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


def test_store_failure_leaves_arguments_intact():
    """Codex P1: a failed store must NOT be replaced by an unredeemable marker."""

    class _BrokenStore:
        def store(self, **kwargs: Any) -> str:
            raise RuntimeError("boom")

    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=_BrokenStore()).apply(messages)
    assert result.compacted_count == 0
    assert result.messages is messages
    assert result.messages[1]["tool_calls"][0]["function"]["arguments"] == LARGE_ARGS


def test_missing_store_leaves_arguments_intact():
    """No store at all == no persistence == no compaction."""
    messages = _openai_conversation()
    result = ToolInputCompactor(_cfg(), compression_store=None).apply(messages)
    assert result.compacted_count == 0
    assert result.messages is messages


def test_store_returning_empty_hash_leaves_arguments_intact():
    class _EmptyHashStore:
        def store(self, **kwargs: Any) -> str:
            return ""

    result = ToolInputCompactor(_cfg(), compression_store=_EmptyHashStore()).apply(
        _openai_conversation()
    )
    assert result.compacted_count == 0


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
                    "function": {"name": "Grep", "arguments": LARGE_ARGS},
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0


# ---------------------------------------------------------------------------
# THE RULE: only reproducible / read-only inputs are compacted (Codex P1).
# A mutating call's result is a bare acknowledgement, so its arguments are the
# sole exact record of the change — and CCR entries expire (default 1,800s).
# ---------------------------------------------------------------------------


def _openai_call(name: str, args: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]


def test_write_tool_input_is_never_compacted():
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Write", MUTATING_ARGS)
    )
    assert result.compacted_count == 0


def test_apply_patch_input_is_never_compacted():
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("apply_patch", json.dumps({"patch": "@@\n+" + "a" * 2000}))
    )
    assert result.compacted_count == 0


def test_mcp_prefixed_write_tool_is_never_compacted():
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("mcp__fs__write_file", MUTATING_ARGS)
    )
    assert result.compacted_count == 0


def test_sql_mutation_input_is_never_compacted():
    sql = "INSERT INTO users (id, name) VALUES " + ",".join(f"({i},'n{i}')" for i in range(200))
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("run_query", json.dumps({"sql": sql}))
    )
    assert result.compacted_count == 0


def test_select_query_input_is_still_compacted():
    sql = (
        "SELECT id, name FROM users WHERE name IN (" + ",".join(f"'n{i}'" for i in range(300)) + ")"
    )
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("run_query", json.dumps({"sql": sql}))
    )
    assert result.compacted_count == 1


def test_shell_heredoc_input_is_never_compacted():
    cmd = "cat <<'EOF' > /tmp/config.yaml\n" + ("key: value\n" * 200) + "EOF"
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Bash", json.dumps({"command": cmd}))
    )
    assert result.compacted_count == 0


def test_shell_redirection_input_is_never_compacted():
    cmd = "printf '%s' '" + "x" * 2000 + "' > /tmp/out.txt"
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Bash", json.dumps({"command": cmd}))
    )
    assert result.compacted_count == 0


def test_read_only_shell_input_is_still_compacted():
    cmd = "rg -n 'def handler' " + " ".join(f"pkg{i}" for i in range(300))
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Bash", json.dumps({"command": cmd}))
    )
    assert result.compacted_count == 1


def test_anthropic_mutating_block_is_never_compacted():
    messages = [
        {"role": "user", "content": "go"},
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
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(messages)
    assert result.compacted_count == 0


def test_arrow_in_a_search_pattern_is_not_mistaken_for_redirection():
    """`->` / `=>` must not read as a write redirection (missed savings)."""
    pattern = "def handler(self) -> dict[str, Any]:  # " + "z" * 1500
    result = ToolInputCompactor(_cfg(), compression_store=_FakeStore()).apply(
        _openai_call("Grep", json.dumps({"pattern": pattern}))
    )
    assert result.compacted_count == 1


def test_is_mutating_tool_input_unit_cases():
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    assert is_mutating_tool_input("Write", "{}") is True
    assert is_mutating_tool_input("apply_patch", "{}") is True
    assert is_mutating_tool_input("Grep", '{"pattern":"a -> b"}') is False
    assert is_mutating_tool_input("Bash", '{"command":"ls -la | wc -l"}') is False
    assert is_mutating_tool_input("Bash", '{"command":"echo hi > out.txt"}') is True
    assert is_mutating_tool_input("Bash", '{"command":"sed -i s/a/b/ f.py"}') is True
    assert is_mutating_tool_input("Bash", '{"command":"git commit -m x"}') is True
    assert is_mutating_tool_input("Bash", '{"command":"git log --oneline -3"}') is False
    assert is_mutating_tool_input("q", '{"sql":"SELECT * FROM t"}') is False
    assert is_mutating_tool_input("q", '{"sql":"DELETE FROM t WHERE id=1"}') is True


def test_unknown_mcp_write_verbs_are_treated_as_mutating():
    """A fixed denylist can't enumerate every MCP server's write ops, so the
    leading verb is the safety net."""
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    for name in (
        "mcp__linear__create_issue",
        "mcp__notion__update_page",
        "mcp__github__push_files",
        "mcp__slack__post_message",
        "upload_artifact",
    ):
        assert is_mutating_tool_input(name, "{}") is True, name


def test_read_only_mcp_tools_are_still_compactable():
    from headroom.transforms.tool_input_compactor import is_mutating_tool_input

    for name in (
        "mcp__github__get_file_contents",
        "mcp__github__search_code",
        "mcp__linear__list_issues",
        "Grep",
        "Read",
        "WebFetch",
    ):
        assert is_mutating_tool_input(name, "{}") is False, name
