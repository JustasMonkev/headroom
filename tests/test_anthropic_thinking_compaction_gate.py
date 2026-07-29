"""F2: the Claude thinking compactor is wired in, behind a DOUBLE gate.

`compact_thinking_to_text()` had zero production callers. It is now invoked
from the Anthropic handler, but only when BOTH hold:

  * `bills_prior_thinking(model)` — older Claude strips prior thinking
    server-side, so compacting there would turn free tokens into billed text.
    Gated specifically at Claude 4.6; models below it still proxy and still get
    ordinary compression, they just skip this transform.
  * `HEADROOM_THINKING_COMPACT` — the transform is LOSSY (signed reasoning
    becomes a generated summary). The billing predicate establishes that
    compaction *could* pay, not that the user accepted a lossy transform.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from headroom.config import model_supports_gated_features
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


def test_thinking_compaction_is_offloaded_to_the_bounded_compression_executor() -> None:
    """It must NOT run inline in the async handler.

    `compact_thinking_to_text()` calls `kompress.compress()` per thinking block:
    local ONNX inference or a blocking `httpx.Client.post()` to the remote
    compressor. Inline, one request stalls the event loop — and therefore every
    other request this worker is serving — for the full inference/network
    duration, with no timeout. It goes through the same bounded executor and the
    same compression timeout as the rest of the pipeline.
    """
    src = inspect.getsource(anthropic_handler.AnthropicHandlerMixin.handle_anthropic_messages)
    assert inspect.iscoroutinefunction(
        anthropic_handler.AnthropicHandlerMixin.handle_anthropic_messages
    )
    idx = src.index("compact_thinking_to_text(")  # the call, not the import
    before = src[max(0, idx - 400) : idx]
    after = src[idx : idx + 600]
    assert "_run_compression_in_executor(" in before, "compaction runs inline on the event loop"
    assert "await" in before, "compaction is not awaited off-loop"
    assert "COMPRESSION_TIMEOUT_SECONDS" in after, "compaction offload has no timeout"


# --------------------------------------------------------------------------
# Gate semantics
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-7",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "anthropic/claude-opus-4-6",
        "claude-opus-5-20260301",
    ],
)
def test_billing_gate_open_at_or_above_claude_4_6(model: str) -> None:
    assert bills_prior_thinking(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
        # Canonical dated Claude 4 ids: the YYYYMMDD stamp is a release date, not
        # a minor version. Reading it as one scored `(4, 20250514)`, opened the
        # gate on a model that strips thinking server-side, and converted free
        # input into billed, lossy text.
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-haiku-20240307",
        "",
        "not-a-model",
    ],
)
def test_billing_gate_closed_below_claude_4_6(model: str) -> None:
    """Fail closed: compacting where thinking is stripped would BILL free tokens."""
    assert bills_prior_thinking(model) is False


def test_thinking_cutoff_does_not_weaken_the_shared_claude_gate() -> None:
    assert bills_prior_thinking("claude-sonnet-4-6") is True
    assert model_supports_gated_features("claude-sonnet-4-6", family="claude") is False


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


# --------------------------------------------------------------------------
# Per-block net-win guard (Codex review, handlers/anthropic.py:2481)
#
# The transform is LOSSY: a signed, server-replayable reasoning block becomes a
# generated summary. The old accept test measured only the SUMMARY against the
# original, so a compressor returning a marginal reduction (40 words -> 39)
# passed — and then the `[prior reasoning, compressed]` marker was prepended,
# pushing the emitted block back over the original. Result: MORE billed input
# tokens AND less information. The accept test must cover the complete emitted
# text, marker included, measured with the request tokenizer.
# --------------------------------------------------------------------------

_MARKER = "[prior reasoning, compressed]"


class _MarginalKompress:
    """Returns a summary one word shorter than the input — a marginal 'win'."""

    def compress(self, text: str, allow_download: bool = True) -> Any:  # noqa: ARG002
        words = text.split()
        return type("R", (), {"compressed": " ".join(words[:-1])})()


def _word_tokenizer(text: str) -> int:
    """Stand-in for the request tokenizer (`tokenizer.count_text`)."""
    return len(text.split())


def _one_thinking_turn(thinking: str) -> list[dict[str, Any]]:
    """A single OLD assistant turn (keep_last_turns=0 compacts it)."""
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": thinking, "signature": "sig-net-win"}],
        },
        {"role": "user", "content": "next"},
    ]


def test_marginal_compaction_is_rejected_and_the_signed_block_survives() -> None:
    """40 words -> 39 words + a 3-word marker is a NET LOSS: keep the original."""
    thinking = " ".join(f"tok{i}" for i in range(40))  # unique text -> no memo reuse
    messages = _one_thinking_turn(thinking)
    original_block = messages[1]["content"][0]

    out, stats = compact_thinking_to_text(
        messages,
        kompress=_MarginalKompress(),
        keep_last_turns=0,
        count_tokens=_word_tokenizer,
    )

    block = out[1]["content"][0]
    assert block["type"] == "thinking", "a lossy replacement that does not shrink was accepted"
    assert block == original_block, "the signed thinking block must survive byte-identical"
    assert block["signature"] == "sig-net-win"
    assert stats["turns_compacted"] == 0 and stats["blocks"] == 0
    assert out[1] is messages[1], "an untouched turn must not be rebuilt"


def test_marker_is_counted_even_without_a_tokenizer() -> None:
    """No tokenizer -> character fallback, still marker-inclusive."""
    thinking = " ".join(f"charfallback{i}" for i in range(40))
    out, stats = compact_thinking_to_text(
        _one_thinking_turn(thinking), kompress=_MarginalKompress(), keep_last_turns=0
    )
    assert out[1]["content"][0]["type"] == "thinking"
    assert stats["blocks"] == 0


def test_a_real_win_is_still_accepted_under_the_tokenizer_guard() -> None:
    """The guard must not disable the transform — a genuine reduction passes."""
    thinking = " ".join(f"realwin{i}" for i in range(80))
    out, stats = compact_thinking_to_text(
        _one_thinking_turn(thinking),
        kompress=_FakeKompress(),  # -> "short summary"
        keep_last_turns=0,
        count_tokens=_word_tokenizer,
    )
    block = out[1]["content"][0]
    assert block == {"type": "text", "text": f"{_MARKER} short summary"}
    assert stats["blocks"] == 1
    assert _word_tokenizer(block["text"]) < _word_tokenizer(thinking)


def test_guard_is_evaluated_per_block_not_per_request() -> None:
    """One winning block must not carry a losing block along with it."""
    win = " ".join(f"perblockwin{i}" for i in range(80))
    lose = " ".join(f"perblocklose{i}" for i in range(40))

    class _Mixed:
        def compress(self, text: str, allow_download: bool = True) -> Any:  # noqa: ARG002
            if text == win:
                return type("R", (), {"compressed": "tiny"})()
            return type("R", (), {"compressed": " ".join(text.split()[:-1])})()

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": win, "signature": "w"},
                {"type": "thinking", "thinking": lose, "signature": "l"},
            ],
        },
        {"role": "user", "content": "next"},
    ]
    losing_block = messages[1]["content"][1]

    out, stats = compact_thinking_to_text(
        messages, kompress=_Mixed(), keep_last_turns=0, count_tokens=_word_tokenizer
    )

    assert out[1]["content"][0]["type"] == "text"
    assert out[1]["content"][1] == losing_block
    assert stats["blocks"] == 1


def test_a_broken_tokenizer_falls_back_instead_of_breaking_the_request() -> None:
    def _boom(text: str) -> int:
        raise RuntimeError("tokenizer exploded")

    thinking = " ".join(f"boom{i}" for i in range(80))
    out, stats = compact_thinking_to_text(
        _one_thinking_turn(thinking),
        kompress=_FakeKompress(),
        keep_last_turns=0,
        count_tokens=_boom,
    )
    assert out[1]["content"][0]["type"] == "text"  # char fallback: a real win still wins
    assert stats["blocks"] == 1


def test_handler_passes_the_request_tokenizer_to_the_compactor() -> None:
    """The guard is only as good as its measure — a word/char proxy is not it."""
    src = inspect.getsource(anthropic_handler.AnthropicHandlerMixin.handle_anthropic_messages)
    idx = src.index("compact_thinking_to_text(")
    call = src[idx : idx + 900]
    assert "count_tokens=tokenizer.count_text" in call, (
        "handler does not hand the request tokenizer to the net-win guard"
    )
