"""Memory tool definitions for LLM function calling.

This module defines the tool specifications in OpenAI function calling format
that allow LLMs to interact with the memory system. These tools enable
autonomous memory management - saving, searching, updating, and deleting
memories as needed during conversations.

Two versions of memory_save are provided:
1. MEMORY_TOOLS - Standard version (backwards compatible)
2. MEMORY_TOOLS_OPTIMIZED - Enhanced version with pre-extraction fields

The optimized version allows the main LLM to extract facts, entities, and
relationships in a single pass, avoiding redundant LLM calls in the storage
backend (Mem0). See extraction.py for the extraction prompts.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Memory Tool Definitions (OpenAI Function Calling Format)
# =============================================================================

# ---------------------------------------------------------------------------
# Shared description strings.
#
# These bytes are replayed on EVERY request of a memory-enabled session
# (sticky tool injection, see proxy/helpers.py:apply_session_sticky_memory_tools),
# so they are deliberately terse — see docs/token-efficiency-review.md A1.
# Keep behavioral guidance the model actually acts on; drop rubrics, DO/DO NOT
# lists and multi-line examples. ``memory_tool_adapter`` imports these so the
# provider-specific variants cannot drift.
# ---------------------------------------------------------------------------

MEMORY_SAVE_DESCRIPTION = (
    "Save durable information to long-term memory: user preferences, personal and "
    "project facts, decisions and their rationale, entity relationships, technical "
    "insights. Do not save transient chat state, secrets, duplicates (search first), "
    "or anything the user asks you not to remember."
)

# The "search memory before searching files" clause used to live in a separate
# ~150-token `## Memory` system block on the Responses/WebSocket path
# (token-efficiency-review A6). It is folded in here instead.
MEMORY_SEARCH_DESCRIPTION = (
    "Recall stored memories. Search memory before searching files or asking the user "
    "— a past session may already hold the answer — and before memory_save to avoid "
    "duplicates."
)

MEMORY_UPDATE_DESCRIPTION = (
    "Correct, extend, or consolidate an existing memory. Use memory_save for genuinely "
    "new facts and memory_delete to remove; edit history is preserved."
)

MEMORY_DELETE_DESCRIPTION = (
    "Remove a memory the user asks you to forget, that is obsolete (merely changed → "
    "use memory_update), or that was saved in error. Soft delete; history is kept."
)

MEMORY_LIST_DESCRIPTION = (
    "List stored memories with their IDs, newest first. Use for browsing, or when you "
    "need an ID and have no good search query; memory_search is the semantic variant."
)

MEMORY_ID_PARAM_DESCRIPTION = (
    "Memory ID: the [id] on a recalled row, or an id from memory_search / memory_list."
)

_IMPORTANCE_DESCRIPTION = "0.0 (low) to 1.0 (critical). Ranks recall order."
_CONTENT_DESCRIPTION = "What to remember; specific and self-contained."
_ENTITIES_DESCRIPTION = "Entity names referenced."
_RELATIONSHIPS_DESCRIPTION = "Entity links as {source, type, target} objects."
_QUERY_DESCRIPTION = "What you are looking for."
_FACTS_DESCRIPTION = "Pre-extracted self-contained facts."
_EXTRACTED_ENTITIES_DESCRIPTION = "Typed entities for graph storage."
_EXTRACTED_RELATIONSHIPS_DESCRIPTION = "Graph links between entities."

# ---------------------------------------------------------------------------
# Item schemas for relationship and pre-extraction arrays.
#
# ``LocalBackend.save_memory`` needs both endpoints to create a relationship;
# ``type`` is optional because it deliberately defaults to ``related_to``.
# Share this object between standard and optimized memory_save so one variant
# cannot silently regress to accepting endpoint-free objects.
# ---------------------------------------------------------------------------
RELATIONSHIP_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "target": {"type": "string"},
        "type": {"type": "string"},
    },
    "required": ["source", "target"],
}

# ---------------------------------------------------------------------------
# The two pre-extraction arrays reach the graph writer directly.
#
# These are NOT decoration and must not be collapsed to a bare
# ``{"type": "object"}`` to save tokens (PR #16 review). ``DirectMem0Adapter
# ._write_graph_to_neo4j`` indexes ``e["entity"]`` / ``e["entity_type"]`` and
# ``rel["source"]`` / ``rel["relationship"]`` / ``rel["destination"]``
# unconditionally, and it runs *after* ``_write_facts_to_qdrant`` has already
# persisted the facts. A schema-valid call missing one of those keys therefore
# raises KeyError on top of a partial save, and the model's natural retry
# duplicates the memories. The required-field constraints are what keep the
# provider from ever emitting such a call.
#
# ``entity_type`` is intentionally left un-enumerated: the graph writer accepts
# any string, so an enum would cost bytes on every request to constrain
# something nothing checks.
# ---------------------------------------------------------------------------
EXTRACTED_ENTITY_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entity": {"type": "string"},
        "entity_type": {"type": "string"},
    },
    "required": ["entity", "entity_type"],
}

EXTRACTED_RELATIONSHIP_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "relationship": {"type": "string"},
        "destination": {"type": "string"},
    },
    "required": ["source", "relationship", "destination"],
}

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": MEMORY_SAVE_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": _CONTENT_DESCRIPTION},
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": _IMPORTANCE_DESCRIPTION,
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _ENTITIES_DESCRIPTION,
                    },
                    "relationships": {
                        "type": "array",
                        "items": RELATIONSHIP_ITEM_SCHEMA,
                        "description": _RELATIONSHIPS_DESCRIPTION,
                    },
                },
                "required": ["content", "importance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": MEMORY_SEARCH_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": _QUERY_DESCRIPTION},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only memories mentioning these entities.",
                    },
                    "include_related": {
                        "type": "boolean",
                        "description": "Also return linked memories.",
                    },
                    "include_scores": {
                        "type": "boolean",
                        "description": "Return relevance scores (default false).",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Max results (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": MEMORY_UPDATE_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": MEMORY_ID_PARAM_DESCRIPTION},
                    "new_content": {
                        "type": "string",
                        "description": "Replacement content; complete and self-contained.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why, for the audit trail (e.g. 'user correction').",
                    },
                },
                "required": ["memory_id", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": MEMORY_DELETE_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": MEMORY_ID_PARAM_DESCRIPTION},
                    "reason": {
                        "type": "string",
                        "description": "Why, for the audit trail (e.g. 'user request').",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": MEMORY_LIST_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max memories to return (default 10, max 100).",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": [],
            },
        },
    },
]


def get_memory_tools() -> list[dict[str, Any]]:
    """Return the list of memory tool definitions.

    Returns:
        List of tool definitions in OpenAI function calling format.
    """
    return MEMORY_TOOLS.copy()


def get_tool_names() -> list[str]:
    """Return the names of all memory tools.

    Returns:
        List of tool names.
    """
    return [tool["function"]["name"] for tool in MEMORY_TOOLS]


# =============================================================================
# Optimized Memory Tools (with pre-extraction support)
# =============================================================================
# These tools include additional fields for pre-extracted facts, entities,
# and relationships. When these fields are provided, the storage backend
# can bypass its internal LLM extraction, resulting in significant speedup.

MEMORY_SAVE_OPTIMIZED: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_save",
        "description": (
            MEMORY_SAVE_DESCRIPTION + " Pre-extracting facts / extracted_entities / "
            "extracted_relationships skips an extraction call in the storage backend."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": _CONTENT_DESCRIPTION},
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": _IMPORTANCE_DESCRIPTION,
                },
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": _FACTS_DESCRIPTION,
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": _ENTITIES_DESCRIPTION,
                },
                "extracted_entities": {
                    "type": "array",
                    "items": EXTRACTED_ENTITY_ITEM_SCHEMA,
                    "description": _EXTRACTED_ENTITIES_DESCRIPTION,
                },
                "relationships": {
                    "type": "array",
                    "items": RELATIONSHIP_ITEM_SCHEMA,
                    "description": _RELATIONSHIPS_DESCRIPTION,
                },
                "extracted_relationships": {
                    "type": "array",
                    "items": EXTRACTED_RELATIONSHIP_ITEM_SCHEMA,
                    "description": _EXTRACTED_RELATIONSHIPS_DESCRIPTION,
                },
                "background": {
                    "type": "boolean",
                    "description": "Save asynchronously; returns a task_id immediately.",
                },
            },
            "required": ["content", "importance"],
        },
    },
}

# Optimized tools list - use this for better performance with DirectMem0Adapter
MEMORY_TOOLS_OPTIMIZED: list[dict[str, Any]] = [
    MEMORY_SAVE_OPTIMIZED,
    MEMORY_TOOLS[1],  # memory_search (unchanged)
    MEMORY_TOOLS[2],  # memory_update (unchanged)
    MEMORY_TOOLS[3],  # memory_delete (unchanged)
    MEMORY_TOOLS[4],  # memory_list (new — chronological browse)
]


def get_memory_tools_optimized() -> list[dict[str, Any]]:
    """Return the optimized memory tool definitions with pre-extraction support.

    Use these tools with DirectMem0Adapter for best performance.
    The main LLM should extract facts/entities/relationships when calling
    memory_save, which bypasses redundant LLM extraction in the backend.

    Returns:
        List of optimized tool definitions in OpenAI function calling format.
    """
    return MEMORY_TOOLS_OPTIMIZED.copy()
