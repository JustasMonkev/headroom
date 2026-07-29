"""Tests for output verbosity steering helpers."""

from __future__ import annotations

from headroom.proxy.output_steering import (
    apply_openai_responses_verbosity_steering,
    apply_verbosity_steering,
    replace_or_append_steering_block,
    steering_text,
)
from headroom.proxy.output_verbosity_policy import STEERING_SENTINEL, STEERING_SUFFIX


def test_replace_or_append_steering_block_replaces_existing_block() -> None:
    old = steering_text(1)
    new = steering_text(3)
    assert old is not None
    assert new is not None
    updated, changed = replace_or_append_steering_block(f"System.\n\n{old}\n\nTail.", new)

    assert changed is True
    assert old not in updated
    assert updated == f"System.\n\n{new}\n\nTail."


def test_anthropic_steering_preserves_cached_prefix_block() -> None:
    cached = {
        "type": "text",
        "text": "Big system prompt.",
        "cache_control": {"type": "ephemeral"},
    }
    body = {"system": [cached.copy()]}

    assert apply_verbosity_steering(body, 2) is True
    assert body["system"][0] == cached
    assert body["system"][1] == {"type": "text", "text": steering_text(2)}


def test_anthropic_steering_tolerates_non_string_system_block_text() -> None:
    # A malformed client block ({"type": "text", "text": null}) must not crash
    # `.startswith` and 500 the request; steering is still appended. The OpenAI
    # chat sibling already guards this exact case.
    body = {
        "system": [
            {"type": "text", "text": None},
            {"type": "text", "text": "Real system prompt."},
        ]
    }

    assert apply_verbosity_steering(body, 2) is True
    # The malformed block is left as-is and a steering block is appended.
    assert body["system"][0] == {"type": "text", "text": None}
    assert body["system"][-1] == {"type": "text", "text": steering_text(2)}


def test_openai_responses_steering_is_idempotent() -> None:
    body = {"instructions": "System."}

    assert apply_openai_responses_verbosity_steering(body, 2) is True
    snapshot = body.copy()
    assert apply_openai_responses_verbosity_steering(body, 2) is False
    assert body == snapshot


def test_openai_chat_steering_appends_to_system_message() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
    }
    assert apply_openai_chat_verbosity_steering(body, 2) is True
    sys_content = body["messages"][0]["content"]
    assert "You are helpful." in sys_content
    assert steering_text(2) in sys_content
    # Other messages and ordering are untouched.
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_openai_chat_steering_is_idempotent_and_swaps_level() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {"messages": [{"role": "system", "content": "S."}]}
    assert apply_openai_chat_verbosity_steering(body, 2) is True
    first = body["messages"][0]["content"]
    # Same level again: no change.
    assert apply_openai_chat_verbosity_steering(body, 2) is False
    assert body["messages"][0]["content"] == first
    # Different level: replace, still exactly one block.
    assert apply_openai_chat_verbosity_steering(body, 4) is True
    swapped = body["messages"][0]["content"]
    assert steering_text(4) in swapped
    assert swapped.count(STEERING_SENTINEL) == 1


def test_openai_chat_steering_inserts_system_when_absent() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert apply_openai_chat_verbosity_steering(body, 3) is True
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == steering_text(3)
    assert body["messages"][1] == {"role": "user", "content": "hi"}


def test_openai_chat_steering_handles_list_content() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {"messages": [{"role": "system", "content": [{"type": "text", "text": "base"}]}]}
    assert apply_openai_chat_verbosity_steering(body, 1) is True
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "base"}
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == steering_text(1)


# ─── Codex P2: structured content must migrate the pre-D4 block ──────────

_LEGACY_BLOCK = (
    "<headroom_output_shaping>\n"
    "Skip preamble and postamble. Do not announce what you are about to "
    "do or recap what you just did; start with the substance.\n"
    "</headroom_output_shaping>"
)


def test_anthropic_system_list_migrates_legacy_block_instead_of_duplicating() -> None:
    """A pre-D4 block echoed back in structured system content must be swapped.

    The structured path only tested ``startswith(STEERING_SENTINEL)``, which
    after D4 means ``<hr_shape>``; the legacy block therefore survived and a
    second block was appended next to it — two conflicting verbosity levels,
    both billed every turn.
    """
    body = {
        "system": [
            {"type": "text", "text": "Big system prompt."},
            {"type": "text", "text": _LEGACY_BLOCK},
        ]
    }

    assert apply_verbosity_steering(body, 4) is True

    assert len(body["system"]) == 2
    assert body["system"][0] == {"type": "text", "text": "Big system prompt."}
    assert body["system"][1] == {"type": "text", "text": steering_text(4)}
    serialized = "".join(b["text"] for b in body["system"])
    assert "headroom_output_shaping" not in serialized
    assert serialized.count(STEERING_SENTINEL) == 1


def test_openai_chat_content_list_migrates_legacy_block_instead_of_duplicating() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "base"},
                    {"type": "text", "text": _LEGACY_BLOCK},
                ],
            }
        ]
    }

    assert apply_openai_chat_verbosity_steering(body, 4) is True

    parts = body["messages"][0]["content"]
    assert len(parts) == 2
    assert parts[0] == {"type": "text", "text": "base"}
    assert parts[1] == {"type": "text", "text": steering_text(4)}
    serialized = "".join(p["text"] for p in parts)
    assert "headroom_output_shaping" not in serialized
    assert serialized.count(STEERING_SENTINEL) == 1


def test_structured_prompts_collapse_legacy_and_current_blocks() -> None:
    body = {
        "system": [
            {"type": "text", "text": _LEGACY_BLOCK},
            {"type": "text", "text": steering_text(1)},
        ]
    }

    assert apply_verbosity_steering(body, 4) is True
    assert body["system"] == [{"type": "text", "text": steering_text(4)}]


def test_structured_prompt_moves_cache_control_to_nearest_retained_predecessor() -> None:
    owner_cache_control = {"type": "ephemeral", "ttl": "5m"}
    duplicate_cache_control = {"type": "ephemeral", "ttl": "1h"}
    body = {
        "system": [
            {
                "type": "text",
                "text": _LEGACY_BLOCK,
                "cache_control": owner_cache_control,
            },
            {"type": "text", "text": "Later retained system context."},
            {
                "type": "text",
                "text": steering_text(1),
                "cache_control": duplicate_cache_control,
            },
        ]
    }

    assert apply_verbosity_steering(body, 4) is True
    assert body["system"] == [
        {
            "type": "text",
            "text": steering_text(4),
            "cache_control": owner_cache_control,
        },
        {
            "type": "text",
            "text": "Later retained system context.",
            "cache_control": duplicate_cache_control,
        },
    ]


def test_structured_prompt_later_cache_control_overwrites_owner_when_adjacent() -> None:
    body = {
        "system": [
            {
                "type": "text",
                "text": _LEGACY_BLOCK,
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            },
            {
                "type": "text",
                "text": steering_text(1),
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ]
    }

    assert apply_verbosity_steering(body, 4) is True
    assert body["system"] == [
        {
            "type": "text",
            "text": steering_text(4),
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]


def test_structured_prompt_moves_each_duplicate_breakpoint_independently() -> None:
    body = {
        "system": [
            {"type": "text", "text": _LEGACY_BLOCK},
            {"type": "text", "text": "first retained"},
            {
                "type": "text",
                "text": steering_text(1),
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            },
            {
                "type": "text",
                "text": "second retained",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            },
            {
                "type": "text",
                "text": _LEGACY_BLOCK,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ]
    }

    assert apply_verbosity_steering(body, 4) is True
    assert body["system"] == [
        {"type": "text", "text": steering_text(4)},
        {
            "type": "text",
            "text": "first retained",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        },
        {
            "type": "text",
            "text": "second retained",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]


def test_structured_prompt_collapses_multiple_blocks_in_one_part() -> None:
    body = {
        "system": [
            {
                "type": "text",
                "text": f"base\n\n{_LEGACY_BLOCK}\n\n{steering_text(1)}\n\ntail",
            }
        ]
    }

    assert apply_verbosity_steering(body, 4) is True
    text = body["system"][0]["text"]
    assert text == f"base\n\n{steering_text(4)}\n\ntail"
    assert text.count(STEERING_SENTINEL) == 1
    assert "headroom_output_shaping" not in text


def test_openai_content_parts_collapse_legacy_and_current_blocks() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": _LEGACY_BLOCK},
                    {"type": "text", "text": steering_text(1)},
                ],
            }
        ]
    }

    assert apply_openai_chat_verbosity_steering(body, 4) is True
    assert body["messages"][0]["content"] == [{"type": "text", "text": steering_text(4)}]


def test_openai_content_parts_preserve_duplicate_breakpoint_position() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": _LEGACY_BLOCK},
                    {"type": "text", "text": "retained"},
                    {
                        "type": "text",
                        "text": steering_text(1),
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    },
                ],
            }
        ]
    }

    assert apply_openai_chat_verbosity_steering(body, 4) is True
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": steering_text(4)},
        {
            "type": "text",
            "text": "retained",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]


def test_structured_steering_ignores_unrelated_hr_shape_tag_in_prompt() -> None:
    """A documented `<hr_shape>` element in a system part is not our block."""
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    # Starts with the sentinel, so the old ``startswith`` test matched it and
    # overwrote the whole part — including the trailing user instruction.
    documented = f"{STEERING_SENTINEL}circle{STEERING_SUFFIX} is the shape element."
    body = {"messages": [{"role": "system", "content": [{"type": "text", "text": documented}]}]}

    assert apply_openai_chat_verbosity_steering(body, 2) is True

    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": documented}
    assert parts[1] == {"type": "text", "text": steering_text(2)}


def test_openai_chat_steering_level_zero_is_noop() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {"messages": [{"role": "system", "content": "S."}]}
    assert apply_openai_chat_verbosity_steering(body, 0) is False
    assert body["messages"][0]["content"] == "S."
