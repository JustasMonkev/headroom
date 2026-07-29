"""Tests for pure output verbosity steering policy."""

from __future__ import annotations

from headroom.proxy.output_verbosity_policy import (
    STEERING_SENTINEL,
    STEERING_SUFFIX,
    contains_steering_block,
    find_steering_block,
    replace_or_append_steering_block,
    steering_text,
)


def test_level_zero_and_unknown_levels_have_no_steering_text() -> None:
    assert steering_text(0) is None
    assert steering_text(99) is None


def test_steering_text_is_wrapped_and_byte_stable() -> None:
    first = steering_text(2)
    second = steering_text(2)
    assert first == second
    assert first is not None
    assert first.startswith(f"{STEERING_SENTINEL}\n")
    assert first.endswith(f"\n{STEERING_SUFFIX}")
    assert "Never restate code" in first


def test_replace_or_append_adds_block_to_nonempty_instructions() -> None:
    block = steering_text(3)
    assert block is not None
    updated, changed = replace_or_append_steering_block("System.", block)
    assert changed is True
    assert updated == f"System.\n\n{block}"


def test_replace_or_append_uses_block_for_empty_instructions() -> None:
    block = steering_text(1)
    assert block is not None
    updated, changed = replace_or_append_steering_block("   ", block)
    assert changed is True
    assert updated == block


def test_replace_or_append_replaces_existing_complete_block_once() -> None:
    old = steering_text(1)
    new = steering_text(4)
    assert old is not None
    assert new is not None
    updated, changed = replace_or_append_steering_block(f"System.\n\n{old}\n\nTail.", new)
    assert changed is True
    assert old not in updated
    assert updated == f"System.\n\n{new}\n\nTail."


def test_unclosed_sentinel_never_truncates_the_prompt() -> None:
    """An unclosed sentinel is NOT our block — the tail must survive verbatim.

    The old behaviour replaced "sentinel .. end of string", so a prompt that
    merely mentioned the tag lost every byte after it. Losing user prompt
    content is strictly worse than paying for a duplicate block.
    """
    new = steering_text(2)
    assert new is not None
    prompt = (
        f"System.\n\n{STEERING_SENTINEL}\nold text without close\n\n"
        "Rules the user really needs: never delete the database."
    )

    updated, changed = replace_or_append_steering_block(prompt, new)

    assert changed is True
    assert updated == f"{prompt}\n\n{new}"
    assert "never delete the database" in updated
    assert "old text without close" in updated


def test_replace_or_append_is_idempotent_when_block_matches() -> None:
    block = steering_text(2)
    assert block is not None
    updated, changed = replace_or_append_steering_block(f"System.\n\n{block}", block)
    assert changed is False
    assert updated == f"System.\n\n{block}"


# ─── D4: short sentinel ─────────────────────────────────────────────────


def test_sentinel_is_short_and_pinned() -> None:
    """The wrapper is pure framing paid for on every steered request.

    `<headroom_output_shaping>` + its closer cost ~16 tok of a ~75-tok level-2
    block. This was a deliberate ONE-TIME cache-busting change — pin the bytes
    so it is not churned again.
    """
    assert STEERING_SENTINEL == "<hr_shape>"
    assert STEERING_SUFFIX == "</hr_shape>"


def test_legacy_sentinel_block_is_replaced_not_duplicated() -> None:
    """A block emitted by a pre-D4 build and echoed back must be swapped."""
    legacy = "<headroom_output_shaping>\nSkip preamble and postamble.\n</headroom_output_shaping>"
    block = steering_text(4)
    assert block is not None

    updated, changed = replace_or_append_steering_block(f"System.\n\n{legacy}\n\nTail.", block)

    assert changed is True
    assert "headroom_output_shaping" not in updated
    assert updated == f"System.\n\n{block}\n\nTail."
    assert updated.count(STEERING_SENTINEL) == 1


# ─── Codex P2: `<hr_shape>` is generic enough to collide with real prompt text ──


def test_unrelated_closed_hr_shape_tag_in_prompt_is_left_alone() -> None:
    """A prompt documenting an XML-ish `<hr_shape>` element is not our block.

    Before verification the delimited text was replaced wholesale, silently
    rewriting content the user wrote.
    """
    block = steering_text(2)
    assert block is not None
    prompt = (
        "When emitting a shape descriptor, use:\n"
        f"{STEERING_SENTINEL}circle r=3{STEERING_SUFFIX}\n"
        "and nothing else."
    )

    updated, changed = replace_or_append_steering_block(prompt, block)

    assert changed is True
    assert "circle r=3" in updated
    assert updated == f"{prompt}\n\n{block}"


def test_real_block_after_an_unrelated_tag_is_still_replaced() -> None:
    """Scanning must not stop at the first (unrelated) sentinel occurrence."""
    old = steering_text(1)
    new = steering_text(4)
    assert old is not None and new is not None
    doc = f"Docs: {STEERING_SENTINEL}example{STEERING_SUFFIX}"
    prompt = f"{doc}\n\n{old}"

    updated, changed = replace_or_append_steering_block(prompt, new)

    assert changed is True
    assert updated == f"{doc}\n\n{new}"
    assert "example" in updated
    assert old not in updated


def test_unrelated_unclosed_tag_does_not_hide_a_real_block() -> None:
    """An unclosed mention earlier in the prompt must not shadow our block."""
    old = steering_text(1)
    new = steering_text(3)
    assert old is not None and new is not None
    prompt = f"Mentioning {STEERING_SENTINEL} bare.\n\n{old}"

    updated, changed = replace_or_append_steering_block(prompt, new)

    assert changed is True
    assert updated == f"Mentioning {STEERING_SENTINEL} bare.\n\n{new}"


def test_legacy_block_needs_a_closing_tag_to_be_replaced() -> None:
    """The no-truncation rule applies to the pre-D4 sentinel too."""
    block = steering_text(2)
    assert block is not None
    prompt = "System.\n\n<headroom_output_shaping>\nunterminated\n\nkeep me"

    updated, changed = replace_or_append_steering_block(prompt, block)

    assert changed is True
    assert updated == f"{prompt}\n\n{block}"
    assert "keep me" in updated


def test_every_emitted_block_is_recognized_as_our_own() -> None:
    """Verification must never reject a block this build actually emits."""
    for level in (1, 2, 3, 4):
        block = steering_text(level)
        assert block is not None
        assert contains_steering_block(block)
        assert find_steering_block(block) == (0, len(block))


def test_levels_are_independent_full_strings_not_cumulative() -> None:
    """The old comment claimed the levels were cumulative; they are not.

    Each level is a standalone instruction — level 4 in particular is a total
    rewrite, not level 3 plus more.
    """
    level_3 = steering_text(3)
    level_4 = steering_text(4)
    assert level_3 is not None and level_4 is not None

    assert "Prefer the smallest edit" in level_3
    assert "Prefer the smallest edit" not in level_4
