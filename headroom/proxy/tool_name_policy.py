"""Tool-definition name extraction policy used by proxy injection helpers."""

from __future__ import annotations

from typing import Any


def extract_tool_name(tool_definition: dict[str, Any]) -> str | None:
    """Extract a stable tool name from a tool definition."""

    name = tool_definition.get("name")
    if isinstance(name, str) and name:
        return name
    function_definition = tool_definition.get("function")
    if isinstance(function_definition, dict):
        function_name = function_definition.get("name")
        if isinstance(function_name, str) and function_name:
            return function_name
    tool_type = tool_definition.get("type")
    if isinstance(tool_type, str) and tool_type:
        return tool_type
    return None


# MCP server labels that Headroom itself registers. Only tools namespaced under
# one of these are treated as "our own" for injection dedup.
#
#   * "headroom"        — headroom/ccr/mcp_server.py, ``Server("headroom")``
#   * "headroom-memory" — headroom/memory/mcp_server.py, ``Server("headroom-memory")``
#   * "headroom_memory" — the Codex TOML key (``[mcp_servers.headroom_memory]``,
#     cli/wrap.py); TOML keys cannot contain "-".
HEADROOM_MCP_SERVER_LABELS: frozenset[str] = frozenset(
    {"headroom", "headroom-memory", "headroom_memory"}
)


def normalize_headroom_tool_name(name: str) -> str:
    """Strip a Headroom-owned ``mcp__<server>__`` namespace prefix, if present.

    ``headroom wrap`` registers Headroom's MCP servers, so the client surfaces
    their tools as ``mcp__headroom__headroom_retrieve`` /
    ``mcp__headroom-memory__memory_save``. Those never compare equal to the
    proxy's own ``headroom_retrieve`` / ``memory_save``, so the injection guards
    used to append a second, near-identical definition — up to the whole
    ~2,600-token memory block twice (A2 in docs/token-efficiency-review.md).

    Only the labels in :data:`HEADROOM_MCP_SERVER_LABELS` are stripped. Stripping
    arbitrary ``mcp__<server>__`` prefixes (or reusing ``config._tool_name_aliases``,
    which is built for broad *exclusion* matching) would be wrong here: an
    unrelated server's ``mcp__other-memory__memory_save`` targets a different
    store, and treating it as ours would suppress Headroom's own tool and
    silently break memory writes.

    Names that are not Headroom-namespaced are returned unchanged.
    """
    if not name:
        return name
    # OpenAI-style ``mcp__server__tool``.
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3 and parts[1] in HEADROOM_MCP_SERVER_LABELS and parts[2]:
            return parts[2]
        return name
    # Single-underscore variant ``mcp_server_tool`` emitted by some
    # Anthropic-speaking clients. Only unambiguous when the label itself has no
    # "_" ("headroom", "headroom-memory"); "headroom_memory" cannot be split
    # reliably here, so it falls through unchanged (no dedup — the safe
    # direction: a duplicate definition costs tokens, a wrong match loses a tool).
    if name.startswith("mcp_"):
        parts = name.split("_", 2)
        if (
            len(parts) == 3
            and parts[1] in HEADROOM_MCP_SERVER_LABELS
            and "_" not in parts[1]
            and parts[2]
        ):
            return parts[2]
    return name
