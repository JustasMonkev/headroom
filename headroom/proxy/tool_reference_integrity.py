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

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Blocks that carry a searchable reference to a deferred tool definition.
_TOOL_REFERENCE_TYPE = "tool_reference"


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
    name = block.get("name")
    return name if isinstance(name, str) and name else None


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

    declared = collect_declared_tool_names(tools)
    if not declared:
        # No tools declared at all: either a tool-free request (nothing to
        # reference) or a shape we don't understand. Either way, pruning every
        # reference would be a bigger change than the bug — leave it alone.
        return messages, set()

    pruned: set[str] = set()
    out: list[dict[str, Any]] = []
    changed = False

    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            out.append(message)
            continue

        new_content: list[Any] = []
        message_changed = False
        for block in content:
            name = _referenced_name(block)
            if name is not None and name not in declared:
                pruned.add(name)
                message_changed = True
                continue
            # A tool_search_tool_result wraps its references one level down.
            nested = block.get("content") if isinstance(block, dict) else None
            if isinstance(nested, list):
                kept_nested = []
                for nested_block in nested:
                    nested_name = _referenced_name(nested_block)
                    if nested_name is not None and nested_name not in declared:
                        pruned.add(nested_name)
                        continue
                    kept_nested.append(nested_block)
                if len(kept_nested) != len(nested):
                    block = copy.deepcopy(block)
                    block["content"] = kept_nested
                    message_changed = True
            new_content.append(block)

        if message_changed:
            changed = True
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
