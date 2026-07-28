"""Tool references in the outgoing request must resolve (issue #7).

Reported symptom, with ``--mode token`` in front of Claude Code::

    API Error: 400 Tool reference 'WaitForMcpServers' not found in available tools

Two independent defects fed it, both covered here:

1. ``inject_tool_search_deferral`` matched its core-tools allowlist
   case-sensitively against a lower_snake_case set, while Claude Code ships
   PascalCase names — so *every* tool got ``defer_loading``, including the core
   read/edit/run loop, maximising the deferred surface that can dangle.
2. Nothing verified that a tool referenced by the conversation is actually
   declared by the request.
"""

from __future__ import annotations

from headroom.proxy.helpers import (
    _TOOL_SEARCH_CORE_TOOLS,
    inject_tool_search_deferral,
)
from headroom.proxy.tool_reference_integrity import (
    collect_declared_tool_names,
    find_dangling_tool_uses,
    prune_dangling_tool_references,
)

# The names Claude Code actually sends (PascalCase), not the lower_snake_case
# spelling the allowlist is written in.
CLAUDE_CODE_CORE = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "Task", "TodoWrite"]


def _tool(name: str) -> dict:
    return {"name": name, "description": f"{name} tool", "input_schema": {"type": "object"}}


def _many_tools() -> list[dict]:
    """Enough tools to clear _TOOL_SEARCH_MIN_TOOLS (12)."""
    mcp = [f"mcp__server__op{i}" for i in range(8)]
    return [_tool(n) for n in [*CLAUDE_CODE_CORE, *mcp]]


class TestCoreToolsAreMatchedCaseInsensitively:
    def test_pascal_case_core_tools_stay_resident(self):
        out = inject_tool_search_deferral(_many_tools())

        deferred = {t["name"] for t in out if isinstance(t, dict) and t.get("defer_loading")}
        for name in CLAUDE_CODE_CORE:
            assert name not in deferred, (
                f"{name} is a core Claude Code tool and must not be deferred — "
                "the allowlist is lower_snake_case and the client is PascalCase"
            )

    def test_non_core_tools_are_still_deferred(self):
        out = inject_tool_search_deferral(_many_tools())

        deferred = {t["name"] for t in out if isinstance(t, dict) and t.get("defer_loading")}
        assert deferred == {f"mcp__server__op{i}" for i in range(8)}

    def test_lower_snake_case_names_still_match(self):
        """The original spelling must keep working."""
        tools = [_tool(n) for n in _TOOL_SEARCH_CORE_TOOLS]
        tools += [_tool(f"other_{i}") for i in range(8)]

        out = inject_tool_search_deferral(tools)

        deferred = {t["name"] for t in out if isinstance(t, dict) and t.get("defer_loading")}
        assert deferred == {f"other_{i}" for i in range(8)}

    def test_multiedit_pascal_case_matches(self):
        tools = [_tool("MultiEdit"), *[_tool(f"x{i}") for i in range(15)]]

        out = inject_tool_search_deferral(tools)

        deferred = {t["name"] for t in out if isinstance(t, dict) and t.get("defer_loading")}
        assert "MultiEdit" not in deferred


class TestDeferralDoesNotMutateCallerTools:
    def test_input_tools_are_not_mutated_when_moving_cache_control(self):
        """The tools list can be a process-global compaction-cache entry.

        Writing cache_control in place leaked one request's breakpoint into
        every later request sharing the same tools digest.
        """
        tools = _many_tools()
        # Client's breakpoint sits on a tool that will be deferred.
        tools[-1]["cache_control"] = {"type": "ephemeral"}
        before = [dict(t) for t in tools]

        out = inject_tool_search_deferral(tools)

        assert tools == before, "inject_tool_search_deferral mutated its input list"
        # The breakpoint was still preserved on a resident tool in the output.
        assert any(
            isinstance(t, dict) and t.get("cache_control") and not t.get("defer_loading")
            for t in out
        )


class TestDeclaredToolNames:
    def test_collects_name_and_type(self):
        declared = collect_declared_tool_names(
            [
                _tool("Bash"),
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "tool_search_tool_regex_20251119"},
                "not-a-dict",
            ]
        )
        assert "Bash" in declared
        assert "web_search" in declared
        assert "tool_search_tool_regex_20251119" in declared

    def test_non_list_is_empty(self):
        assert collect_declared_tool_names(None) == set()


class TestPruneDanglingToolReferences:
    def _messages_with_reference(self, name: str) -> list[dict]:
        return [
            {"role": "user", "content": "find me a tool"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": {
                            "type": "tool_search_tool_search_result",
                            "tool_references": [
                                {"type": "tool_reference", "tool_name": name},
                                {"type": "tool_reference", "tool_name": "Bash"},
                            ],
                        },
                    }
                ],
            },
        ]

    def test_dangling_nested_reference_is_pruned(self):
        """The exact reported shape: a replayed reference to a vanished tool."""
        messages = self._messages_with_reference("WaitForMcpServers")

        out, pruned = prune_dangling_tool_references(messages, [_tool("Bash")])

        assert pruned == {"WaitForMcpServers"}
        kept = out[1]["content"][0]["content"]["tool_references"]
        assert [b["tool_name"] for b in kept] == ["Bash"]

    def test_top_level_dangling_reference_is_pruned(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_reference", "tool_name": "WaitForMcpServers"},
                    {"type": "text", "text": "hello"},
                ],
            }
        ]

        out, pruned = prune_dangling_tool_references(messages, [_tool("Bash")])

        assert pruned == {"WaitForMcpServers"}
        assert [b["type"] for b in out[0]["content"]] == ["text"]

    def test_resolvable_references_are_untouched(self):
        """No unresolvable reference → same object back, no prefix perturbation."""
        messages = self._messages_with_reference("WaitForMcpServers")
        tools = [_tool("Bash"), _tool("WaitForMcpServers")]

        out, pruned = prune_dangling_tool_references(messages, tools)

        assert pruned == set()
        assert out is messages

    def test_input_messages_are_not_mutated(self):
        messages = self._messages_with_reference("WaitForMcpServers")

        out, pruned = prune_dangling_tool_references(messages, [_tool("Bash")])

        assert pruned
        assert out is not messages
        original_references = messages[1]["content"][0]["content"]["tool_references"]
        assert len(original_references) == 2, "input was mutated"

    def test_explicitly_empty_tools_prunes_all_references(self):
        """An understood empty tool surface makes every reference dangling."""
        messages = self._messages_with_reference("WaitForMcpServers")

        out, pruned = prune_dangling_tool_references(messages, [])

        assert pruned == {"Bash", "WaitForMcpServers"}
        result = out[1]["content"][0]["content"]
        assert result["tool_references"] == []
        assert result["type"] == "tool_search_tool_search_result"

    def test_empty_custom_tool_result_content_gets_valid_placeholder(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_search",
                        "content": [
                            {
                                "type": "tool_reference",
                                "tool_name": "WaitForMcpServers",
                            }
                        ],
                    }
                ],
            }
        ]

        out, pruned = prune_dangling_tool_references(messages, [_tool("Bash")])

        assert pruned == {"WaitForMcpServers"}
        repaired = out[0]["content"][0]["content"]
        assert repaired and repaired[0]["type"] == "text"
        assert repaired[0]["text"]

    def test_empty_message_content_gets_valid_placeholder(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_reference",
                        "tool_name": "WaitForMcpServers",
                    }
                ],
            }
        ]

        out, pruned = prune_dangling_tool_references(messages, [_tool("Bash")])

        assert pruned == {"WaitForMcpServers"}
        assert out[0]["content"] and out[0]["content"][0]["type"] == "text"

    def test_absent_tools_shape_is_a_noop(self):
        """Missing tools is ambiguous, so preserve the byte-stable history."""
        messages = self._messages_with_reference("WaitForMcpServers")

        out, pruned = prune_dangling_tool_references(messages, None)

        assert pruned == set()
        assert out is messages

    def test_unrecognized_nonempty_tools_shape_is_a_noop(self):
        messages = self._messages_with_reference("WaitForMcpServers")

        out, pruned = prune_dangling_tool_references(messages, [{"unknown": True}])

        assert pruned == set()
        assert out is messages

    def test_tool_use_is_never_pruned(self):
        """Dropping a tool_use would orphan its paired tool_result."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Gone", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        ]

        out, pruned = prune_dangling_tool_references(messages, [_tool("Bash")])

        assert pruned == set()
        assert out is messages
        # …but it is detectable, so a caller can skip the prefix replay.
        assert find_dangling_tool_uses(messages, {"Bash"}) == {"Gone"}

    def test_string_content_messages_pass_through(self):
        messages = [{"role": "user", "content": "plain text"}]
        out, pruned = prune_dangling_tool_references(messages, [_tool("Bash")])
        assert pruned == set()
        assert out is messages
