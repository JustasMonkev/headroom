"""Codex P2: tool-input compaction must be OFF in lossless mode.

`--lossless` suppresses every CCR marker AND the `headroom_retrieve` tool
injection. Tool-input compaction replaces arguments with a `hash=` marker that
is only redeemable through that tool, so running both together strands the
arguments — a lossy outcome in the mode whose entire contract is losslessness.
"""

from __future__ import annotations

from headroom.config import ToolInputCompactionConfig
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


def test_router_lossless_disables_tool_input_compaction() -> None:
    cfg = ContentRouterConfig(
        lossless=True,
        tool_input_compaction=ToolInputCompactionConfig(enabled=True),
    )
    router = ContentRouter(cfg)
    assert router.config.tool_input_compaction.enabled is False
    # The rest of the lossless invariant is untouched.
    assert router.config.ccr_inject_marker is False
    assert router.config.smart_crusher_lossless_only is True


def test_router_non_lossless_keeps_tool_input_compaction() -> None:
    cfg = ContentRouterConfig(
        lossless=False,
        tool_input_compaction=ToolInputCompactionConfig(enabled=True),
    )
    router = ContentRouter(cfg)
    assert router.config.tool_input_compaction.enabled is True


def test_lossless_router_never_compacts_tool_inputs_end_to_end() -> None:
    """With both flags set the arguments must survive the pipeline intact."""
    from headroom.tokenizer import Tokenizer

    class _Counter:
        def count_text(self, text: str) -> int:
            return max(1, len(text) // 4)

        def count_messages(self, messages: list[dict]) -> int:
            return sum(self.count_text(str(m.get("content", ""))) for m in messages)

    big_args = {"pattern": "x", "glob": "y" * 3000}
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Grep", "input": big_args}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "sure"},
    ]

    router = ContentRouter(
        ContentRouterConfig(
            lossless=True,
            tool_input_compaction=ToolInputCompactionConfig(
                enabled=True, min_chars=100, protect_recent_turns=0
            ),
        )
    )
    result = router.apply(
        messages,
        Tokenizer(_Counter(), "test-model"),  # type: ignore[arg-type]
        model_limit=100_000,
    )
    block = result.messages[1]["content"][0]
    assert block["input"] == big_args
    assert not any(t.startswith("tool_input_compaction") for t in result.transforms_applied)
