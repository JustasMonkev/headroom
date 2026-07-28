"""Keep the outgoing request's tool references resolvable (issue #7).

Anthropic rejects a request whose conversation references a tool that the
request does not declare::

    400 Tool reference 'WaitForMcpServers' not found in available tools

Nothing in the proxy used to check that invariant. It is easy to violate
without any single stage being wrong:

* ``overlay_cached_prefix`` replays the previously-*forwarded* prefix
  byte-identical (``headroom/cache/prefix_tracker.py``) with no coupling to the
  current ``tools`` array. That replay is the cache-safety centerpiece of token
  mode, so token mode reinstates old ``tool_search_tool_result`` blocks
  constantly.
* Claude Code's tool surface is not stable across turns — ``WaitForMcpServers``
  is exposed while MCP servers are still connecting and then disappears.

Combine the two and a replayed ``tool_reference`` names a tool the current turn
no longer declares. The request is well-formed by every local invariant and
still 400s.

This module prunes exactly those unresolvable ``tool_reference`` blocks. It
deliberately does **not** touch ``tool_use``: dropping one would orphan its
paired ``tool_result`` and trade a clear 400 for a confusing one. A dangling
``tool_use`` is reported to the caller instead so it can decide (skipping the
prefix replay is the usual answer).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Blocks that carry a searchable reference to a deferred tool definition.
_TOOL_REFERENCE_TYPE = "tool_reference"
_EMPTY_REFERENCE_REPAIR_TEXT = {
    "type": "text",
    "text": "[Unavailable tool reference removed]",
}
_REMOVE = object()


def collect_declared_tool_names(tools: Any) -> set[str]:
    """Names the outgoing request actually declares.

    Server/typed tools (``web_search``, ``computer``, the tool-search tool …)
    carry a ``type`` and may omit ``name``; both keys are collected so a
    reference by either resolves.
    """
    declared: set[str] = set()
    if not isinstance(tools, list):
        return declared
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        for key in ("name", "type"):
            value = tool.get(key)
            if isinstance(value, str) and value:
                declared.add(value)
    return declared


def _referenced_name(block: Any) -> str | None:
    if not isinstance(block, dict) or block.get("type") != _TOOL_REFERENCE_TYPE:
        return None
    # Anthropic's wire format is ``tool_name``. Keep ``name`` as a
    # compatibility fallback for older/custom producers.
    name = block.get("tool_name", block.get("name"))
    return name if isinstance(name, str) and name else None


def _prune_reference_value(
    value: Any,
    declared: set[str],
    pruned: set[str],
) -> tuple[Any, bool]:
    """Copy-on-write pruning for the two Anthropic reference container shapes.

    Built-in tool search nests references under
    ``content.tool_references``; custom tool search can put them in a
    ``tool_result.content`` list. The latter cannot be left empty because
    Anthropic rejects empty content arrays, so replace an emptied ``content``
    list with a neutral text block. An empty ``tool_references`` array is valid
    and is the documented "no matches" shape.
    """
    name = _referenced_name(value)
    if name is not None:
        if name not in declared:
            pruned.add(name)
            return _REMOVE, True
        return value, False

    if isinstance(value, list):
        changed = False
        out: list[Any] = []
        for item in value:
            new_item, item_changed = _prune_reference_value(item, declared, pruned)
            changed = changed or item_changed
            if new_item is not _REMOVE:
                out.append(new_item)
        return (out, True) if changed else (value, False)

    if not isinstance(value, dict):
        return value, False

    changed = False
    out_dict = value
    for key in ("content", "tool_references"):
        child = value.get(key)
        if not isinstance(child, (dict, list)):
            continue
        new_child, child_changed = _prune_reference_value(child, declared, pruned)
        if not child_changed:
            continue
        if not changed:
            out_dict = dict(value)
            changed = True
        if new_child is _REMOVE:
            new_child = []
        if key == "content" and isinstance(new_child, list) and not new_child:
            # ``content: []`` is invalid for both messages and tool_result
            # blocks. Preserve a syntactically valid, non-actionable marker.
            new_child = [dict(_EMPTY_REFERENCE_REPAIR_TEXT)]
        out_dict[key] = new_child
    return out_dict, changed


def find_dangling_tool_uses(messages: list[dict[str, Any]], declared: set[str]) -> set[str]:
    """Tool names invoked by a ``tool_use`` block but not declared."""
    dangling: set[str] = set()
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if isinstance(name, str) and name and name not in declared:
                dangling.add(name)
    return dangling


def prune_dangling_tool_references(
    messages: list[dict[str, Any]],
    tools: Any,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Drop ``tool_reference`` blocks naming tools this request does not declare.

    Returns ``(messages, pruned_names)``. When nothing is unresolvable the input
    list is returned unchanged (same object) so the common path stays free — no
    copy, and no perturbation of the byte-stable prefix the provider cached.
    """
    if not isinstance(messages, list) or not messages:
        return messages, set()

    if not isinstance(tools, list):
        # Absent or unknown top-level shape: do not guess that the request is
        # tool-free.
        return messages, set()
    declared = collect_declared_tool_names(tools)
    if tools and not declared:
        # A non-empty list with no understood declarations is likewise an
        # unknown shape. An explicitly empty list is different: it is a known
        # declaration of zero tools, so every tool_reference must be pruned.
        return messages, set()

    pruned: set[str] = set()
    out: list[dict[str, Any]] = []
    changed = False

    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            out.append(message)
            continue

        new_content, message_changed = _prune_reference_value(content, declared, pruned)

        if message_changed:
            changed = True
            if not new_content:
                new_content = [dict(_EMPTY_REFERENCE_REPAIR_TEXT)]
            new_message = dict(message)
            new_message["content"] = new_content
            out.append(new_message)
        else:
            out.append(message)

    if not changed:
        return messages, set()

    logger.warning(
        "Pruned %d unresolvable tool_reference(s) from the outgoing request: %s. "
        "The conversation referenced tools this turn no longer declares (the "
        "client's tool surface changed between turns); forwarding them would "
        "have produced a 400 'not found in available tools'.",
        len(pruned),
        ", ".join(sorted(pruned)),
    )
    return out, pruned
