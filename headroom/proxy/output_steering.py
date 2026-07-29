"""Byte-stable output verbosity steering helpers."""

from __future__ import annotations

from typing import Any

from headroom.proxy.output_verbosity_policy import (
    contains_steering_block,
    replace_or_append_steering_block,
    steering_text,
)


def _resteer_content_parts(
    parts: list[Any], key: str, text: str, *, text_parts_only: bool = False
) -> bool | None:
    """Collapse structured steering blocks to one current block.

    Structured system content used to be matched with
    ``startswith(STEERING_SENTINEL)``, which after D4 shortened the sentinel
    meant these paths silently stopped recognizing pre-D4
    ``<headroom_output_shaping>`` blocks: the stale block stayed and a second
    one was appended beside it, so two conflicting verbosity levels were sent —
    and paid for — on every turn. Going through
    :func:`replace_or_append_steering_block` gives the structured paths the
    same legacy migration the plain-string path already had.
    """
    found = False
    changed = False
    kept: list[Any] = []
    for part in parts:
        value = part.get(key) if isinstance(part, dict) else None
        if (
            (text_parts_only and isinstance(part, dict) and part.get("type") != "text")
            or not isinstance(value, str)
            or not contains_steering_block(value)
        ):
            kept.append(part)
            continue
        updated, part_changed = replace_or_append_steering_block(value, text if not found else "")
        found = True
        changed = changed or part_changed
        if updated or len(part) > 2:
            if part_changed:
                part[key] = updated
            kept.append(part)
        else:
            changed = True
    if not found:
        return None
    if len(kept) != len(parts):
        parts[:] = kept
    return changed


def apply_verbosity_steering(body: dict[str, Any], level: int) -> bool:
    """Append the steering block to the tail of the Anthropic system prompt.

    Appending after the last system block keeps any ``cache_control``
    breakpoint on an earlier block intact: the cached prefix is unchanged and
    only the small, byte-stable steering block is reprocessed.
    """
    text = steering_text(level)
    if text is None:
        return False

    system = body.get("system")
    if system is None:
        body["system"] = [{"type": "text", "text": text}]
        return True
    if isinstance(system, str):
        body["system"] = [
            {"type": "text", "text": system},
            {"type": "text", "text": text},
        ]
        return True
    if isinstance(system, list):
        changed = _resteer_content_parts(system, "text", text)
        if changed is not None:
            return changed
        system.append({"type": "text", "text": text})
        return True
    return False


def apply_openai_chat_verbosity_steering(
    body: dict[str, Any],
    level: int,
) -> bool:
    """Append or replace the steering block in an OpenAI chat/completions body.

    OpenAI ``/v1/chat/completions`` carries the system prompt as a
    ``role: "system"`` (or ``"developer"``) message inside ``messages`` rather
    than a top-level field, so it needs its own injector (the Anthropic
    ``system`` and Responses ``instructions`` variants do not reach it — the
    root cause of GitHub Copilot CLI seeing zero output savings, #2302).

    The block is appended to the tail of the last system/developer message so a
    treatment conversation's steering stays byte-stable across turns (and
    re-applies idempotently via the sentinel). When the request carries no
    system message at all, one is inserted at the front. Returns True only when
    the body actually changed.
    """
    text = steering_text(level)
    if text is None:
        return False

    messages = body.get("messages")
    if not isinstance(messages, list):
        return False

    target: dict[str, Any] | None = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") in ("system", "developer"):
            target = message
    if target is None:
        # No system prompt to append to — insert one carrying just the block.
        messages.insert(0, {"role": "system", "content": text})
        return True

    content = target.get("content")
    if content is None:
        target["content"] = text
        return True
    if isinstance(content, str):
        updated, changed = replace_or_append_steering_block(content, text)
        if changed:
            target["content"] = updated
        return changed
    if isinstance(content, list):
        # OpenAI also accepts a content-part list ([{"type": "text", ...}]).
        part_changed = _resteer_content_parts(content, "text", text, text_parts_only=True)
        if part_changed is not None:
            return part_changed
        content.append({"type": "text", "text": text})
        return True
    return False


def apply_openai_responses_verbosity_steering(
    body: dict[str, Any],
    level: int,
) -> bool:
    """Append or replace steering in OpenAI Responses ``instructions``."""
    text = steering_text(level)
    if text is None:
        return False

    instructions = body.get("instructions")
    if instructions is None:
        body["instructions"] = text
        return True
    if not isinstance(instructions, str):
        return False

    updated, changed = replace_or_append_steering_block(instructions, text)
    if changed:
        body["instructions"] = updated
    return changed
