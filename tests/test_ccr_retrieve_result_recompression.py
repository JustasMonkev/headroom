"""A retrieved original must never be lossily re-compressed on the next turn.

The model calls ``headroom_retrieve`` because it needs the full content. If the
proxy's ContentRouter then lossy-compresses that tool result, the model gets a
fresh CCR marker for the same bytes instead of the content — and must retrieve
again, paying for the round trip every time (a retrieval loop).

SmartCrusher.apply() already skipped the bare ``headroom_retrieve`` name
(#1077); the ContentRouter path — the one the proxy runs for Anthropic and
OpenAI Chat — had no guard at all, for either spelling.
"""

from __future__ import annotations

import json
import random

import pytest

from headroom.compress import compress
from headroom.config import DEFAULT_EXCLUDE_TOOLS
from headroom.parser import CCR_RETRIEVAL_MARKER_RE
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

pytest.importorskip("headroom._core", reason="Rust core required for SmartCrusher")

RETRIEVE_NAMES = ["headroom_retrieve", "mcp__headroom__headroom_retrieve"]


def _retrieved_log() -> str:
    rng = random.Random(2)
    return "\n".join(
        f"2026-09-27 10:{i // 60 % 60:02d}:{i % 60:02d} "
        f"{rng.choice(['INFO'] * 8 + ['WARN', 'ERROR'])} [worker-{rng.randint(1, 8)}] "
        f"handler: request id={rng.getrandbits(40):x} path=/api/items/{rng.randint(1, 5000)} "
        f"latency_ms={rng.randint(1, 900)}"
        for i in range(400)
    )


def _legacy_mcp_payload() -> str:
    """The pre-raw-text MCP retrieve result: one huge JSON string field, which
    SmartCrusher's opaque-blob path replaced with a `<<ccr:…>>` marker."""
    return json.dumps(
        {
            "hash": "a1b2c3d4e5f6a1b2c3d4e5f6",
            "source": "local",
            "original_content": _retrieved_log(),
        },
        separators=(",", ":"),
    )


def _anthropic(tool_name: str, content: str) -> list[dict]:
    return [
        {"role": "user", "content": "find the errors"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": {"hash": "a1b2"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": content}],
        },
    ]


def _openai(tool_name: str, content: str) -> list[dict]:
    return [
        {"role": "user", "content": "find the errors"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": '{"hash":"a1b2"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": content},
    ]


def _anthropic_out(result) -> str:
    return result.messages[-1]["content"][0]["content"]


def _openai_out(result) -> str:
    return result.messages[-1]["content"]


@pytest.mark.parametrize("tool_name", RETRIEVE_NAMES)
@pytest.mark.parametrize(
    ("build", "extract", "model"),
    [(_anthropic, _anthropic_out, "claude-sonnet-4-5-20250929"), (_openai, _openai_out, "gpt-4o")],
)
def test_retrieve_result_gets_no_new_marker(tool_name, build, extract, model) -> None:
    payload = _legacy_mcp_payload()

    out = extract(compress(build(tool_name, payload), model=model))

    assert not CCR_RETRIEVAL_MARKER_RE.search(out)
    assert json.loads(out)["original_content"] == _retrieved_log()


def test_same_payload_from_another_tool_is_still_compressed() -> None:
    """Control: the guard is keyed on the tool, not on the content."""
    payload = _legacy_mcp_payload()

    out = _anthropic_out(compress(_anthropic("Bash", payload), model="claude-sonnet-4-5-20250929"))

    assert out != payload


def test_retrieve_results_keep_lossless_folds() -> None:
    """Excluded, not verbatim: pretty JSON is still minified (data-lossless)."""
    rows = [{"id": i, "name": f"user_{i}", "status": "ok"} for i in range(200)]
    pretty = json.dumps(rows, indent=2)

    result = compress(_anthropic("mcp__headroom__headroom_retrieve", pretty))
    out = _anthropic_out(result)

    assert json.loads(out) == rows
    assert len(out) < len(pretty)
    assert result.tokens_after < result.tokens_before


def test_guard_survives_a_caller_exclude_set_that_replaces_defaults() -> None:
    """ContentRouterConfig.exclude_tools replaces DEFAULT_EXCLUDE_TOOLS; the
    retrieve guard must not depend on the caller remembering it."""
    assert "headroom_retrieve" in DEFAULT_EXCLUDE_TOOLS
    router = ContentRouter(ContentRouterConfig(exclude_tools={"Read"}))
    from headroom import OpenAIProvider, Tokenizer

    tokenizer = Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")
    payload = _legacy_mcp_payload()

    result = router.apply(_anthropic("mcp__headroom__headroom_retrieve", payload), tokenizer)

    assert not CCR_RETRIEVAL_MARKER_RE.search(_anthropic_out(result))


@pytest.mark.parametrize("shape", ["anthropic", "openai"])
def test_old_retrieve_result_is_not_age_decayed_into_a_marker(shape) -> None:
    """Excluded tools age out of protection past the recent window (the
    proxy's default token mode sets protect_recent_reads_fraction=0.3). A
    retrieve result must stay exempt from lossy compression at any age, or a
    long conversation re-mints a marker for content the model already fetched."""
    from headroom import OpenAIProvider, Tokenizer

    router = ContentRouter(ContentRouterConfig(protect_recent_reads_fraction=0.3))
    tokenizer = Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")
    payload = _legacy_mcp_payload()
    build = _anthropic if shape == "anthropic" else _openai
    messages = build("mcp__headroom__headroom_retrieve", payload)
    # Push the retrieve result far outside the recent-message window.
    for i in range(40):
        messages.append({"role": "assistant", "content": f"step {i}"})
        messages.append({"role": "user", "content": f"continue {i}"})
    idx = 2

    result = router.apply(messages, tokenizer)

    retrieved = result.messages[idx]
    out = retrieved["content"][0]["content"] if shape == "anthropic" else retrieved["content"]
    assert not CCR_RETRIEVAL_MARKER_RE.search(out)
    assert json.loads(out)["original_content"] == _retrieved_log()
