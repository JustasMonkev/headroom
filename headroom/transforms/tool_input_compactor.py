"""Compact completed tool-call inputs (arguments) via CCR.

Tool OUTPUTS are compressed by ContentRouter; tool-call INPUTS are not.
Historical Write payloads, apply_patch bodies, shell heredocs, and SQL
strings therefore stay verbatim in context for the rest of the session
even though the model already acted on their results. This pre-processing
pass replaces large, completed tool-call arguments with a compact marker
+ CCR hash, preserving the call id and tool name so provider validation
and conversation structure are untouched. Handles both wire shapes:

- OpenAI: ``message["tool_calls"][i]["function"]["arguments"]`` (JSON string)
- Anthropic: ``tool_use`` content blocks' ``input`` (object)

The original serialized arguments are stored in the CCR compression store
under the marker's hash, so ``headroom_retrieve`` / ``/v1/retrieve/{hash}``
recovers the exact bytes on demand.

Safety rules (each prevents a concrete failure mode):
- Only calls whose matching tool result appears in a LATER message are
  compacted — a pending call's arguments are live working context.
- The trailing ``protect_recent_turns`` assistant messages are never
  touched: the model frequently reuses recent arguments (e.g. iterating
  on a patch), and re-deriving them from a retrieval round-trip would
  cost more than the compaction saves.
- Messages inside the provider's frozen cache prefix are never mutated —
  rewriting cached bytes busts the prefix cache, which costs more than
  the token savings.
- Already-compacted inputs (the ``_ccr`` key) are skipped, so repeated
  passes over the same conversation are idempotent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import ToolInputCompactionConfig

logger = logging.getLogger(__name__)

# Replacement key carrying the marker inside the compacted arguments.
CCR_INPUT_KEY = "_ccr"


@dataclass
class ToolInputCompactionResult:
    """Output of the tool-input compaction pass."""

    messages: list[dict[str, Any]]
    compacted_count: int = 0
    chars_before: int = 0
    chars_after: int = 0
    transforms_applied: list[str] = field(default_factory=list)
    ccr_hashes: list[str] = field(default_factory=list)


class ToolInputCompactor:
    """Replace large completed tool-call arguments with CCR markers.

    Mirrors :class:`ReadLifecycleManager`'s shape: a pre-processing pass over
    ``messages`` run at the top of ``ContentRouter.apply``, storing originals
    in the shared compression store.
    """

    def __init__(
        self,
        config: ToolInputCompactionConfig,
        compression_store: Any | None = None,
    ):
        self.config = config
        self.store = compression_store

    def apply(
        self,
        messages: list[dict[str, Any]],
        frozen_message_count: int = 0,
    ) -> ToolInputCompactionResult:
        """Compact completed tool-call arguments in place-safe copies.

        Args:
            messages: Conversation messages (OpenAI or Anthropic shape).
            frozen_message_count: Leading messages inside the provider's
                prefix cache; never mutated.
        """
        result = ToolInputCompactionResult(messages=messages)
        if not self.config.enabled or not messages:
            return result

        completed_at = self._result_message_indices(messages)
        if not completed_at:
            return result

        protected = self._protected_assistant_indices(messages)

        new_messages: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            if idx < frozen_message_count or idx in protected:
                continue
            compacted = self._compact_assistant_message(idx, msg, completed_at, result)
            if compacted is not None:
                if new_messages is None:
                    new_messages = list(messages)
                new_messages[idx] = compacted

        if new_messages is not None:
            result.messages = new_messages
            logger.info(
                "ToolInputCompactor: compacted %d tool inputs, %d -> %d chars",
                result.compacted_count,
                result.chars_before,
                result.chars_after,
            )
        return result

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _result_message_indices(self, messages: list[dict[str, Any]]) -> dict[str, int]:
        """Map tool_call_id -> index of the message carrying its result."""
        indices: dict[str, int] = {}
        for idx, msg in enumerate(messages):
            # OpenAI: role "tool" messages reference tool_call_id.
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if isinstance(tc_id, str) and tc_id and tc_id not in indices:
                    indices[tc_id] = idx
                continue
            # Anthropic: user messages carry tool_result content blocks.
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tc_id = block.get("tool_use_id")
                if isinstance(tc_id, str) and tc_id and tc_id not in indices:
                    indices[tc_id] = idx
        return indices

    def _protected_assistant_indices(self, messages: list[dict[str, Any]]) -> set[int]:
        """Indices of the trailing N assistant messages (never compacted)."""
        keep = self.config.protect_recent_turns
        if keep <= 0:
            return set()
        assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        return set(assistant_indices[-keep:])

    # ------------------------------------------------------------------
    # Replacement
    # ------------------------------------------------------------------

    def _compact_assistant_message(
        self,
        msg_index: int,
        msg: dict[str, Any],
        completed_at: dict[str, int],
        result: ToolInputCompactionResult,
    ) -> dict[str, Any] | None:
        """Return a compacted copy of ``msg``, or None if nothing changed."""
        changed = False
        new_msg = dict(msg)

        # OpenAI shape: tool_calls array with JSON-string arguments.
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            new_calls: list[Any] = []
            for tc in tool_calls:
                replacement = self._compact_openai_call(msg_index, tc, completed_at, result)
                if replacement is not None:
                    new_calls.append(replacement)
                    changed = True
                else:
                    new_calls.append(tc)
            if changed:
                new_msg["tool_calls"] = new_calls

        # Anthropic shape: tool_use content blocks with object input.
        content = msg.get("content")
        if isinstance(content, list):
            new_blocks: list[Any] = []
            block_changed = False
            for block in content:
                replacement = self._compact_anthropic_block(msg_index, block, completed_at, result)
                if replacement is not None:
                    new_blocks.append(replacement)
                    block_changed = True
                else:
                    new_blocks.append(block)
            if block_changed:
                new_msg["content"] = new_blocks
                changed = True

        return new_msg if changed else None

    def _compact_openai_call(
        self,
        msg_index: int,
        tc: Any,
        completed_at: dict[str, int],
        result: ToolInputCompactionResult,
    ) -> dict[str, Any] | None:
        if not isinstance(tc, dict):
            return None
        tc_id = tc.get("id")
        func = tc.get("function")
        if not isinstance(tc_id, str) or not isinstance(func, dict):
            return None
        args = func.get("arguments")
        if not isinstance(args, str) or len(args) < self.config.min_chars:
            return None
        if completed_at.get(tc_id, -1) <= msg_index:
            return None  # Pending or same-message result: arguments are live.
        if CCR_INPUT_KEY in args[:16]:
            return None  # Already compacted (idempotence).

        marker, ccr_hash = self._store_original(
            args, tool_name=str(func.get("name", "")), tool_call_id=tc_id
        )
        replacement_args = json.dumps({CCR_INPUT_KEY: marker}, separators=(",", ":"))
        self._record(result, str(func.get("name", "")), len(args), len(replacement_args), ccr_hash)
        return {**tc, "function": {**func, "arguments": replacement_args}}

    def _compact_anthropic_block(
        self,
        msg_index: int,
        block: Any,
        completed_at: dict[str, int],
        result: ToolInputCompactionResult,
    ) -> dict[str, Any] | None:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            return None
        tc_id = block.get("id")
        inp = block.get("input")
        if not isinstance(tc_id, str) or not isinstance(inp, dict):
            return None
        if CCR_INPUT_KEY in inp:
            return None  # Already compacted (idempotence).
        try:
            serialized = json.dumps(inp, separators=(",", ":"), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return None
        if len(serialized) < self.config.min_chars:
            return None
        if completed_at.get(tc_id, -1) <= msg_index:
            return None  # Pending or same-message result: arguments are live.

        marker, ccr_hash = self._store_original(
            serialized, tool_name=str(block.get("name", "")), tool_call_id=tc_id
        )
        self._record(result, str(block.get("name", "")), len(serialized), len(marker), ccr_hash)
        return {**block, "input": {CCR_INPUT_KEY: marker}}

    def _store_original(
        self, serialized: str, *, tool_name: str, tool_call_id: str
    ) -> tuple[str, str]:
        """Persist the original arguments; return (marker, ccr_hash).

        Best-effort CCR persistence (mirrors read_lifecycle): a store failure
        must not break compress() — the marker still carries a valid hash for
        deployments where the store catches up later.
        """
        ccr_hash = hashlib.sha256(serialized.encode()).hexdigest()[:24]
        if self.store is not None:
            try:
                ccr_hash = self.store.store(
                    original=serialized,
                    compressed="",
                    tool_name=tool_name or "tool",
                    tool_call_id=tool_call_id,
                    compression_strategy="tool_input_compaction",
                    explicit_hash=ccr_hash,
                )
            except Exception as e:  # noqa: BLE001 - storage failure must not break the request
                logger.warning(
                    "tool_input_compaction: CCR store failed for %s: %s", tool_call_id, e
                )
        # NOTE: the literal phrase "Retrieve original: hash=" is load-bearing —
        # the hash collectors in ccr/tool_injection.py match it, which keeps the
        # headroom_retrieve tool injected while compacted inputs are in context.
        marker = f"[tool input elided. Retrieve original: hash={ccr_hash}]"
        return marker, ccr_hash

    @staticmethod
    def _record(
        result: ToolInputCompactionResult,
        tool_name: str,
        chars_before: int,
        chars_after: int,
        ccr_hash: str,
    ) -> None:
        result.compacted_count += 1
        result.chars_before += chars_before
        result.chars_after += chars_after
        result.transforms_applied.append(f"tool_input_compaction:{tool_name or 'tool'}")
        result.ccr_hashes.append(ccr_hash)
