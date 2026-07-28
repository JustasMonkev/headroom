"""Token budget + contract guards for the injected memory tool schemas.

The memory tool definitions are *sticky*: once a session injects them they are
replayed byte-for-byte on every subsequent request (see
``apply_session_sticky_memory_tools``). That is the right prompt-cache
tradeoff, which means the only lever is making the bytes smaller — the A1
finding in docs/token-efficiency-review.md.

These tests pin what the shrink must not break:

* a hard ceiling on the serialized size, so a future "just one more bullet"
  edit shows up as a failing test rather than as ~1,400 extra tokens on every
  request of every memory-enabled session;
* the parameters that are consumed in production (``reason`` on update/delete
  is recorded in edit history and forwarded to audit metadata — Codex review
  P2 explicitly asked for it to be kept);
* ``include_scores``, which the handler reads but which was missing from every
  injected schema (Codex review P2);
* the one novel clause from the deleted inline ``## Memory`` block (A6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from headroom.memory.tools import (
    MEMORY_SEARCH_DESCRIPTION,
    MEMORY_TOOLS,
    MEMORY_TOOLS_OPTIMIZED,
)
from headroom.proxy.memory_tool_adapter import (
    ANTHROPIC_CUSTOM_TOOLS,
    GEMINI_TOOLS,
    OPENAI_TOOLS,
)

# Serialized-character ceilings. Chars are used here so the assertion needs no
# tokenizer (and no network to fetch a BPE file); roughly 4 chars/token for
# this JSON. Measured with cl100k_base at the time of the A1 shrink:
#
#   MEMORY_TOOLS            9,625 chars / 2,208 tok -> 3,300 chars /   836 tok
#   MEMORY_TOOLS_OPTIMIZED 10,217 chars / 2,356 tok -> 3,923 chars /   984 tok
#   ANTHROPIC_CUSTOM_TOOLS  3,774 chars /   922 tok -> 3,093 chars /   778 tok
#   OPENAI_TOOLS            2,959 chars /   742 tok -> 3,221 chars /   813 tok
#   GEMINI_TOOLS            2,225 chars /   534 tok -> 2,734 chars /   672 tok
#
# (The adapter's OpenAI/Gemini variants grew slightly: they used to carry
# one-line descriptions with no usage guidance at all, and now share the same
# strings as every other variant. The adapter has no production caller — the
# proxy path is MEMORY_TOOLS_OPTIMIZED — so consistency wins there.)
#
# The ceilings leave ~15% headroom for wording tweaks; a genuine feature that
# needs more should raise them deliberately, with the new number recorded.
_CEILINGS: dict[str, tuple[list[dict[str, Any]], int]] = {
    "MEMORY_TOOLS": (MEMORY_TOOLS, 3_900),
    "MEMORY_TOOLS_OPTIMIZED": (MEMORY_TOOLS_OPTIMIZED, 4_500),
    "ANTHROPIC_CUSTOM_TOOLS": (ANTHROPIC_CUSTOM_TOOLS, 3_600),
    "OPENAI_TOOLS": (OPENAI_TOOLS, 3_800),
    "GEMINI_TOOLS": (GEMINI_TOOLS, 3_200),
}


def _by_name(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for t in tools:
        if t.get("name") == name:
            return t
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name") == name:
            return t
    raise AssertionError(f"{name} not found")


def _params(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function")
    if isinstance(fn, dict):
        return dict(fn.get("parameters") or {})
    return dict(tool.get("input_schema") or tool.get("parameters") or {})


ALL_VARIANTS = pytest.mark.parametrize(
    "tools",
    [MEMORY_TOOLS, MEMORY_TOOLS_OPTIMIZED, ANTHROPIC_CUSTOM_TOOLS, OPENAI_TOOLS, GEMINI_TOOLS],
    ids=["standard", "optimized", "anthropic", "openai", "gemini"],
)


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "tools", "ceiling"), [(k, v[0], v[1]) for k, v in _CEILINGS.items()]
)
def test_serialized_schema_stays_under_ceiling(
    label: str, tools: list[dict[str, Any]], ceiling: int
) -> None:
    size = len(json.dumps(tools, ensure_ascii=False))
    assert size <= ceiling, (
        f"{label} serializes to {size} chars (ceiling {ceiling}). These bytes are "
        "replayed on EVERY request of a memory-enabled session — see "
        "docs/token-efficiency-review.md A1 before raising this."
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@ALL_VARIANTS
def test_memory_search_exposes_include_scores(tools: list[dict[str, Any]]) -> None:
    """``MemoryHandler._execute_search`` reads ``include_scores`` and omits
    per-row scores unless it is set. A parameter the handler honours but no
    schema advertises is unreachable (Codex review P2)."""
    props = _params(_by_name(tools, "memory_search"))["properties"]
    assert "include_scores" in props
    assert props["include_scores"]["type"] == "boolean"


@ALL_VARIANTS
def test_reason_params_are_kept(tools: list[dict[str, Any]]) -> None:
    """``reason`` is consumed in production: ``_execute_update`` records it in
    the edit-history entry and forwards it into the backend audit trail.
    Dropping it to save tokens would silently degrade the audit trail."""
    for name in ("memory_update", "memory_delete"):
        props = _params(_by_name(tools, name))["properties"]
        assert "reason" in props, f"{name} lost its reason parameter"
        assert props["reason"]["type"] == "string"


@ALL_VARIANTS
def test_required_params_unchanged(tools: list[dict[str, Any]]) -> None:
    """The shrink is description-only: required-parameter contracts hold."""
    assert set(_params(_by_name(tools, "memory_save"))["required"]) == {
        "content",
        "importance",
    }
    assert _params(_by_name(tools, "memory_search"))["required"] == ["query"]
    assert set(_params(_by_name(tools, "memory_update"))["required"]) == {
        "memory_id",
        "new_content",
    }
    assert _params(_by_name(tools, "memory_delete"))["required"] == ["memory_id"]


@ALL_VARIANTS
def test_search_description_is_shared_across_variants(tools: list[dict[str, Any]]) -> None:
    """Provider variants import the same strings, so they cannot drift (the
    old copies had already diverged into five different lengths)."""
    tool = _by_name(tools, "memory_search")
    desc = (tool.get("function") or tool).get("description")
    assert desc == MEMORY_SEARCH_DESCRIPTION


def test_search_description_carries_the_search_before_files_clause() -> None:
    """A6: the deleted inline ``## Memory`` block had exactly one clause that
    the tool descriptions did not already cover. It lives here now."""
    lowered = MEMORY_SEARCH_DESCRIPTION.lower()
    assert "search memory before searching files" in lowered


def test_inline_memory_instruction_block_is_gone() -> None:
    """A6: the ~150-token ``## Memory`` block on the Responses/WebSocket path
    duplicated tool-description guidance and mutated ``instructions`` (which
    is otherwise byte-stable, i.e. prompt-cache safe)."""
    src = Path("headroom/proxy/handlers/openai.py").read_text(encoding="utf-8")
    assert "## Memory\\n" not in src
    assert "You have persistent memory via memory_search" not in src


# ---------------------------------------------------------------------------
# include_scores behaviour (the schema addition has to actually do something)
# ---------------------------------------------------------------------------


class _ScoreResult:
    def __init__(self, mid: str, content: str, score: float) -> None:
        self.memory = type("M", (), {"id": mid, "content": content})()
        self.score = score
        self.related_entities: list[str] = []


class _ScoreBackend:
    async def search_memories(self, **_: Any) -> list[_ScoreResult]:
        return [_ScoreResult("id-1", "a fact", 0.87654)]


def _search(handler_input: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

    handler = MemoryHandler(MemoryConfig(enabled=True, backend="local"))
    handler._backend = _ScoreBackend()  # type: ignore[assignment]
    handler._initialized = True
    return json.loads(asyncio.run(handler._execute_search(handler_input, "u")))


def test_search_omits_scores_by_default() -> None:
    payload = _search({"query": "x"})
    assert "score" not in payload["memories"][0]


def test_include_scores_true_returns_scores() -> None:
    payload = _search({"query": "x", "include_scores": True})
    assert payload["memories"][0]["score"] == pytest.approx(0.877)


def test_adapter_search_honours_include_scores() -> None:
    import asyncio

    from headroom.proxy.memory_tool_adapter import (
        MemoryToolAdapter,
        MemoryToolAdapterConfig,
    )

    adapter = MemoryToolAdapter(MemoryToolAdapterConfig(enabled=True))
    adapter._backend = _ScoreBackend()  # type: ignore[assignment]
    adapter._initialized = True

    off = json.loads(asyncio.run(adapter._execute_search({"query": "x"}, "u")))
    assert "score" not in off["memories"][0]

    on = json.loads(
        asyncio.run(adapter._execute_search({"query": "x", "include_scores": True}, "u"))
    )
    assert on["memories"][0]["score"] == pytest.approx(0.877)
