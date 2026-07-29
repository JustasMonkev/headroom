"""B1: the `<headroom:tool_digest sha256=…>` marker is out-of-band.

The digest is provenance metadata with no production consumer — a repo-wide
grep for `tool_digest` / `extract_markers` finds only `headroom/utils.py`
(which builds and parses it) and `tests/test_utils.py`. In message content it
cost ~16-18 tokens on EVERY crushed tool result, billed to the model, read by
nothing. It now travels in `TransformResult.markers_inserted` instead.
"""

from __future__ import annotations

import json

import pytest

from headroom import OpenAIProvider, Tokenizer


def _build_extension() -> None:
    try:
        from headroom._core import SmartCrusher  # noqa: F401
    except ImportError:
        pytest.skip(
            "headroom._core not built — run `bash scripts/build_rust_extension.sh`",
            allow_module_level=True,
        )


_build_extension()


def _tokenizer() -> Tokenizer:
    provider = OpenAIProvider()
    return Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")


def _crusher():
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    return SmartCrusher(SmartCrusherConfig(min_tokens_to_crush=10))


_PAYLOAD = json.dumps({"items": [{"id": i, "v": "x" * 10} for i in range(40)]})


def _openai_messages():
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "query", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": _PAYLOAD},
    ]


def _anthropic_messages():
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "u1", "name": "query", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "u1", "content": _PAYLOAD}],
        },
    ]


def test_openai_crushed_content_carries_no_digest_marker():
    result = _crusher().apply(_openai_messages(), _tokenizer())

    content = result.messages[1]["content"]
    assert content != _PAYLOAD, "payload should have been crushed"
    assert "headroom:tool_digest" not in content
    assert not content.endswith("\n")


def test_anthropic_crushed_content_carries_no_digest_marker():
    result = _crusher().apply(_anthropic_messages(), _tokenizer())

    content = result.messages[1]["content"][0]["content"]
    assert content != _PAYLOAD, "payload should have been crushed"
    assert "headroom:tool_digest" not in content


@pytest.mark.parametrize("build", [_openai_messages, _anthropic_messages])
def test_digest_still_reported_out_of_band(build):
    """Dropping it from content must not drop it from the transform result."""
    result = _crusher().apply(build(), _tokenizer())

    assert result.markers_inserted, "digest should still be reported"
    assert all(m.startswith("<headroom:tool_digest sha256=") for m in result.markers_inserted)


def test_digest_removal_shows_up_as_fewer_tokens():
    from headroom.utils import compute_short_hash, create_tool_digest_marker

    tok = _tokenizer()
    marker = create_tool_digest_marker(compute_short_hash(_PAYLOAD))

    # The marker is not free: this is exactly what every crushed message used
    # to pay on top of its content.
    assert tok.count_text(marker) >= 10
