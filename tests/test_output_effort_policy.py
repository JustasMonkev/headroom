"""Tests for pure output effort policy decisions."""

from __future__ import annotations

from headroom.proxy.output_effort_policy import (
    LEGACY_THINKING_FLOOR,
    can_create_openai_text_verbosity,
    clamp_legacy_thinking_budget,
    lower_effort_value,
    lower_text_verbosity_value,
)


def test_lower_effort_value_lowers_known_higher_effort_to_target() -> None:
    assert lower_effort_value("xhigh", "low") == "low"
    assert lower_effort_value("max", "medium") == "medium"


def test_lower_effort_value_keeps_lower_equal_unknown_or_non_string_values() -> None:
    assert lower_effort_value("low", "medium") is None
    assert lower_effort_value("medium", "medium") is None
    assert lower_effort_value("turbo", "low") is None
    assert lower_effort_value("high", "turbo") is None
    assert lower_effort_value(None, "low") is None


def test_clamp_legacy_thinking_budget_only_clamps_enabled_over_floor() -> None:
    assert (
        clamp_legacy_thinking_budget(
            thinking_type="enabled",
            budget_tokens=32_000,
        )
        == LEGACY_THINKING_FLOOR
    )
    assert (
        clamp_legacy_thinking_budget(
            thinking_type="enabled",
            budget_tokens=LEGACY_THINKING_FLOOR,
        )
        is None
    )
    assert clamp_legacy_thinking_budget(thinking_type="adaptive", budget_tokens=32_000) is None
    assert clamp_legacy_thinking_budget(thinking_type="enabled", budget_tokens="32000") is None


def test_can_create_openai_text_verbosity_only_at_or_above_the_feature_cutoff() -> None:
    """Native output controls are gated on MIN_GPT_FEATURE_VERSION (gpt >= 5.5)."""
    assert can_create_openai_text_verbosity("gpt-5.5", first_party_target=True)
    assert can_create_openai_text_verbosity("GPT-5.5-codex", first_party_target=True)
    assert can_create_openai_text_verbosity("openai/gpt-6", first_party_target=True)
    # Below the cutoff the model still works — it just falls back to the
    # portable instruction-steering lever instead of the native knob.
    assert not can_create_openai_text_verbosity("gpt-5", first_party_target=True)
    assert not can_create_openai_text_verbosity("GPT-5.1", first_party_target=True)
    assert not can_create_openai_text_verbosity("gpt-5.4", first_party_target=True)
    assert not can_create_openai_text_verbosity("gpt-4o", first_party_target=True)
    assert not can_create_openai_text_verbosity("o3", first_party_target=True)
    assert not can_create_openai_text_verbosity(None, first_party_target=True)


def test_can_create_openai_text_verbosity_requires_a_first_party_upstream() -> None:
    """The upstream gate is ANDed with the model gate, and defaults closed.

    ``text.verbosity`` is an OpenAI-Responses-native field. A vendor-prefixed
    id like ``openai/gpt-5.5`` clears the model cutoff but is exactly the id
    shape a compatible gateway (reached via ``x-headroom-base-url``) uses, and
    such a gateway may reject the field outright.
    """
    for model in ("gpt-5.5", "GPT-5.5-codex", "openai/gpt-6"):
        assert can_create_openai_text_verbosity(model, first_party_target=True) is True
        assert can_create_openai_text_verbosity(model, first_party_target=False) is False
        # Unknown upstream must fail closed rather than fail open.
        assert can_create_openai_text_verbosity(model) is False


def test_lower_text_verbosity_value_lowers_existing_verbose_values() -> None:
    assert lower_text_verbosity_value("medium") == "low"
    assert lower_text_verbosity_value("high") == "low"
    assert lower_text_verbosity_value("low") is None
    assert lower_text_verbosity_value("chatty") is None
    assert lower_text_verbosity_value(None) is None
