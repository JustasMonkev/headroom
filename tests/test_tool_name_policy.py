from __future__ import annotations

from headroom.proxy.helpers import _extract_tool_name, apply_session_sticky_ccr_tool
from headroom.proxy.tool_name_policy import extract_tool_name, normalize_headroom_tool_name


def test_extracts_anthropic_custom_tool_name() -> None:
    assert extract_tool_name({"name": "memory_save"}) == "memory_save"


def test_extracts_openai_function_tool_name() -> None:
    assert (
        extract_tool_name({"type": "function", "function": {"name": "memory_search"}})
        == "memory_search"
    )


def test_extracts_native_tool_type_when_name_absent() -> None:
    assert extract_tool_name({"type": "memory_20250818"}) == "memory_20250818"


def test_prefers_explicit_name_over_function_and_type() -> None:
    assert (
        extract_tool_name(
            {
                "name": "headroom_retrieve",
                "type": "function",
                "function": {"name": "memory_save"},
            }
        )
        == "headroom_retrieve"
    )


def test_ignores_empty_or_non_string_names() -> None:
    assert extract_tool_name({"name": "", "function": {"name": ""}, "type": ""}) is None
    assert extract_tool_name({"name": 123, "function": {"name": 456}, "type": []}) is None


def test_helpers_private_wrapper_keeps_existing_import_path() -> None:
    tool_definition = {"function": {"name": "memory_update"}}

    assert _extract_tool_name(tool_definition) == extract_tool_name(tool_definition)


# ─── A2: Headroom-owned MCP prefix normalization ───────────────────────


def test_normalizes_headroom_owned_mcp_prefixes() -> None:
    """`headroom wrap` registers our own MCP servers, so the client surfaces
    our tools namespaced. Those must compare equal to the bare names the
    proxy injects, or we append a second copy of every definition."""
    assert normalize_headroom_tool_name("mcp__headroom__headroom_retrieve") == "headroom_retrieve"
    assert normalize_headroom_tool_name("mcp__headroom__headroom_compress") == "headroom_compress"
    assert normalize_headroom_tool_name("mcp__headroom-memory__memory_save") == "memory_save"
    # Anthropic-style single-underscore wrapper, where the label is unambiguous.
    assert normalize_headroom_tool_name("mcp_headroom_headroom_retrieve") == "headroom_retrieve"
    assert normalize_headroom_tool_name("mcp_headroom-memory_memory_save") == "memory_save"


def test_does_not_normalize_foreign_mcp_servers() -> None:
    """A same-named tool from an unrelated MCP server is a DIFFERENT tool.

    Stripping arbitrary `mcp__<server>__` prefixes would let e.g.
    `mcp__other-memory__memory_save` (a different store) suppress Headroom's
    own `memory_save` injection and silently break memory writes.
    """
    for foreign in (
        "mcp__other-memory__memory_save",
        "mcp__supermemory__memory_save",
        "mcp__someserver__headroom_retrieve",
        "mcp_othermemory_memory_save",
    ):
        assert normalize_headroom_tool_name(foreign) == foreign


def test_leaves_unnamespaced_and_degenerate_names_alone() -> None:
    assert normalize_headroom_tool_name("headroom_retrieve") == "headroom_retrieve"
    assert normalize_headroom_tool_name("Read") == "Read"
    assert normalize_headroom_tool_name("") == ""
    assert normalize_headroom_tool_name("mcp__headroom__") == "mcp__headroom__"
    assert normalize_headroom_tool_name("mcp__headroom") == "mcp__headroom"


def test_headroom_ccr_tool_dedupes_against_namespaced_mcp_registration() -> None:
    """End-to-end A2: the sticky-CCR injector must not double up."""
    existing = [{"name": "mcp__headroom__headroom_retrieve", "input_schema": {}}]

    tools_out, injected = apply_session_sticky_ccr_tool(
        provider="anthropic",
        session_id=None,
        request_id="req-a2",
        existing_tools=existing,
        has_compressed_content_this_turn=True,
    )

    assert injected is False
    assert len(tools_out) == 1


def test_foreign_mcp_retrieve_tool_does_not_suppress_injection() -> None:
    """Negative case: an unrelated server's `headroom_retrieve` is not ours."""
    existing = [{"name": "mcp__someserver__headroom_retrieve", "input_schema": {}}]

    tools_out, injected = apply_session_sticky_ccr_tool(
        provider="anthropic",
        session_id=None,
        request_id="req-a2-neg",
        existing_tools=existing,
        has_compressed_content_this_turn=True,
    )

    assert injected is True
    assert len(tools_out) == 2
    assert tools_out[-1]["name"] == "headroom_retrieve"
