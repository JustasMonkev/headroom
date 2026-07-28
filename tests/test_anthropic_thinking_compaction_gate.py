"""F2: the Claude thinking compactor is wired in, behind a DOUBLE gate.

`compact_thinking_to_text()` had zero production callers. It is now invoked
from the Anthropic handler, but only when BOTH hold:

  * `bills_prior_thinking(model)` — pre-4.6 Claude strips prior thinking
    server-side, so compacting there would turn free tokens into billed text.
  * `HEADROOM_THINKING_COMPACT` — the transform is LOSSY (signed reasoning
    becomes a generated summary). The billing predicate establishes that
    compaction *could* pay, not that the user accepted a lossy transform.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from headroom.proxy.handlers import anthropic as anthropic_handler
from headroom.transforms.thinking_compactor import (
    bills_prior_thinking,
    compact_thinking_to_text,
)

LONG = " ".join(["reasoning"] * 80)


class _FakeKompress:
    def compress(self, text: str, allow_download: bool = True) -> Any:  # noqa: ARG002
        return type("R", (), {"compressed": "short summary"})()


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": LONG, "signature": "s1"},
                {"type": "tool_use", "id": "t1", "name": "Grep", "input": {"pattern": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": LONG, "signature": "s2"}],
        },
    ]


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------
def test_handler_calls_the_thinking_compactor() -> None:
    """The transform must have a production caller (it had none)."""
    src = inspect.getsource(anthropic_handler)
    assert "compact_thinking_to_text" in src


def test_handler_gates_on_both_the_billing_predicate_and_the_opt_in() -> None:
    src = inspect.getsource(anthropic_handler)
    idx = src.index("compact_thinking_to_text")
    window = src[max(0, idx - 3000) : idx]
    assert "HEADROOM_THINKING_COMPACT" in window, "opt-in gate missing"
    assert "bills_prior_thinking" in window, "billing gate missing"


# --------------------------------------------------------------------------
# Gate semantics
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-5"])
def test_billing_gate_open_for_46_plus(model: str) -> None:
    assert bills_prior_thinking(model) is True


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "claude-3-5-sonnet-20241022"],
)
def test_billing_gate_closed_for_pre_46(model: str) -> None:
    assert bills_prior_thinking(model) is False


# --------------------------------------------------------------------------
# Behaviour the handler relies on
# --------------------------------------------------------------------------
def test_latest_turn_thinking_is_preserved() -> None:
    out, stats = compact_thinking_to_text(_messages(), kompress=_FakeKompress(), keep_last_turns=1)
    assert out[1]["content"][0]["type"] == "text"
    assert out[1]["content"][1]["type"] == "tool_use", "tool_use blocks must survive"
    assert out[3]["content"][0]["type"] == "thinking", "latest turn must keep its thinking"
    assert stats["turns_compacted"] == 1


def test_compaction_is_byte_stable_across_turns() -> None:
    """Cache safety: the same thinking must map to the same bytes every turn."""
    k = _FakeKompress()
    first, _ = compact_thinking_to_text(_messages(), kompress=k, keep_last_turns=1)
    second, _ = compact_thinking_to_text(_messages(), kompress=k, keep_last_turns=1)
    assert first[1]["content"][0] == second[1]["content"][0]


def test_no_kompress_is_a_noop() -> None:
    messages = _messages()
    out, stats = compact_thinking_to_text(messages, kompress=None, keep_last_turns=1)
    assert out is messages
    assert stats["turns_compacted"] == 0
