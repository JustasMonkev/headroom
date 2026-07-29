"""The shared model-specific-feature cutoff.

Headroom's ordinary compression runs on every model. A handful of transforms
are different: they exploit provider behaviour that only exists on recent
frontier models (server-side tool search, native output controls, prior-turn
thinking being re-billed as input). Those engage only at or above
``MIN_CLAUDE_FEATURE_VERSION`` / ``MIN_GPT_FEATURE_VERSION``.

Two properties are load-bearing and pinned here:

* **One definition.** Every gate routes through
  :func:`model_supports_gated_features`, so raising a cutoff is a one-line
  change and no call site can drift.
* **Fail closed.** An unrecognized or unparseable model id must NOT get the
  gated features. Firing one where the provider doesn't support it costs real
  tokens (compacting thinking a pre-cutoff model would have stripped for free)
  or 400s the request (``defer_loading`` / ``text.verbosity``). A false
  negative only costs a missed saving.
"""

from __future__ import annotations

import pytest

from headroom.config import (
    MIN_CLAUDE_FEATURE_VERSION,
    MIN_GPT_FEATURE_VERSION,
    model_supports_gated_features,
    parse_model_family_version,
)


def test_cutoffs_are_the_agreed_versions() -> None:
    assert MIN_CLAUDE_FEATURE_VERSION == (4, 8)
    assert MIN_GPT_FEATURE_VERSION == (5, 5)


# (model id, gated features engage?)
GATE_TABLE: list[tuple[object, bool]] = [
    # --- Claude: below the 4.8 cutoff. No carve-outs — 4.5/4.6/4.7 all lose it.
    ("claude-opus-4-6", False),
    ("claude-sonnet-4-6", False),
    ("claude-sonnet-4-7", False),
    ("claude-sonnet-4-5", False),
    ("claude-sonnet-4-5-20250929", False),
    ("claude-haiku-4-5", False),
    ("claude-haiku-4-5-20251001", False),
    ("claude-sonnet-4", False),
    ("claude-3-5-sonnet", False),
    ("claude-3-5-sonnet-20241022", False),
    ("claude-3-opus-20240229", False),
    # --- Dated canonical ids: the YYYYMMDD stamp is a DATE, not a minor
    # version. `claude-sonnet-4-20250514` is Claude 4 Sonnet -> below the
    # cutoff. Reading the date as the minor made it (4, 20250514) and let
    # exactly the models this cutoff excludes through the gate.
    ("claude-sonnet-4-20250514", False),
    ("claude-opus-4-20250514", False),
    ("anthropic/claude-sonnet-4-20250514", False),
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", False),
    ("claude-opus-4-1-20250805", False),
    ("claude-3-haiku-20240307", False),
    # --- Split `YYYY-MM-DD` snapshot stamps are dates too. OpenAI spells them
    # with separators, so `gpt-5-2025-08-07` tokenizes to 5/2025/08/07 and the
    # 8-digit rule alone read `2025` as the minor, clearing the 5.5 cutoff for
    # an old GPT-5 snapshot.
    ("gpt-5-2025-08-07", False),
    ("openai/gpt-5-2025-08-07", False),
    ("gpt-5.4-2026-02-01", False),
    ("claude-sonnet-4-2025-05-14", False),
    # --- Claude: at/above the cutoff.
    ("claude-opus-4-8", True),
    ("claude-opus-4-8-20260210", True),
    ("claude-sonnet-4-8-20260210", True),
    ("claude-opus-4-9", True),
    ("claude-sonnet-5", True),
    ("claude-opus-5-1", True),
    ("claude-opus-6", True),
    # --- Vendor-prefixed / gateway forms must parse identically.
    ("anthropic/claude-opus-4-8", True),
    ("anthropic/claude-opus-4-6", False),
    ("us.anthropic.claude-opus-4-8-v1:0", True),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", False),
    ("claude-opus-4-8[1m]", True),  # the wrap `[1m]` context suffix
    ("CLAUDE-OPUS-4-8", True),  # case-insensitive
    # --- GPT: below the 5.5 cutoff.
    ("gpt-5", False),
    ("gpt-5.1", False),
    ("gpt-5.2", False),
    ("gpt-5.3", False),
    ("gpt-5.4", False),
    ("gpt-5.4-2026-02-01", False),
    ("gpt-5-codex", False),
    ("gpt-4o", False),
    ("gpt-4.1", False),
    # --- GPT: at/above the cutoff.
    ("gpt-5.5", True),
    ("gpt-5.5-codex", True),
    ("gpt-5.6-terra", True),
    ("gpt-6", True),
    ("gpt-6.2", True),
    ("openai/gpt-5.5", True),
    ("openai/gpt-5", False),
    # --- Families with no version-gated features at all.
    ("o1", False),
    ("o3", False),
    ("gemini-2.5-pro", False),
    # --- Fail closed on garbage.
    ("", False),
    ("   ", False),
    ("garbage", False),
    ("claude", False),  # family but no version
    ("some-future-model", False),
    (None, False),
    (12345, False),
    (object(), False),
]


@pytest.mark.parametrize(("model", "expected"), GATE_TABLE, ids=lambda v: str(v)[:48])
def test_model_supports_gated_features(model: object, expected: bool) -> None:
    assert model_supports_gated_features(model) is expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-8", ("claude", (4, 8))),
        ("claude-sonnet-4-5-20250929", ("claude", (4, 5))),
        ("claude-3-5-sonnet-20241022", ("claude", (3, 5))),
        ("us.anthropic.claude-opus-4-6-v1:0", ("claude", (4, 6))),
        ("anthropic/claude-opus-4-8", ("claude", (4, 8))),
        ("claude-sonnet-5", ("claude", (5, 0))),
        # A YYYYMMDD stamp terminates the version run instead of becoming the
        # minor: these are Claude 4 / Claude 3, not "Claude 4.20250514".
        ("claude-sonnet-4-20250514", ("claude", (4, 0))),
        ("claude-opus-4-20250514", ("claude", (4, 0))),
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", ("claude", (4, 0))),
        ("claude-opus-4-8-20260210", ("claude", (4, 8))),
        ("claude-opus-4-1-20250805", ("claude", (4, 1))),
        ("claude-3-haiku-20240307", ("claude", (3, 0))),
        ("claude-3-opus-20240229", ("claude", (3, 0))),
        ("gpt-5", ("gpt", (5, 0))),
        ("gpt-5.5-codex", ("gpt", (5, 5))),
        ("gpt-5.4-2026-02-01", ("gpt", (5, 4))),
        # A split YYYY-MM-DD stamp is a date, not a minor: an OpenAI snapshot
        # id is the base model plus a release date.
        ("gpt-5-2025-08-07", ("gpt", (5, 0))),
        ("openai/gpt-5-2025-08-07", ("gpt", (5, 0))),
        ("gpt-5.5-2026-01-15", ("gpt", (5, 5))),
        ("gpt-6-2026-03-01", ("gpt", (6, 0))),
        ("claude-sonnet-4-2025-05-14", ("claude", (4, 0))),
        # ... but a non-date digit run keeps its version meaning.
        ("gpt-5-12345678", ("gpt", (5, 12345678))),
        ("openai/gpt-6.2", ("gpt", (6, 2))),
        # Not a plausible calendar date -> still a version component, so the
        # date rule stays narrow and can't silently swallow real minors.
        ("gpt-6-12345678", ("gpt", (6, 12345678))),
        ("gpt-4o", None),  # no parseable version -> unknown, not a guess
        ("o3", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_model_family_version(
    model: object, expected: tuple[str, tuple[int, int]] | None
) -> None:
    assert parse_model_family_version(model) == expected


def test_family_restriction_blocks_cross_vendor_ids() -> None:
    """A provider-specific gate must not be opened by the other vendor's model."""
    assert model_supports_gated_features("gpt-6", family="gpt") is True
    assert model_supports_gated_features("gpt-6", family="claude") is False
    assert model_supports_gated_features("claude-sonnet-5", family="claude") is True
    assert model_supports_gated_features("claude-sonnet-5", family="gpt") is False


def test_every_gate_agrees_with_the_shared_predicate() -> None:
    """No gate may re-derive its own threshold."""
    from headroom.proxy.helpers import (
        _model_supports_anthropic_tool_search,
        _model_supports_openai_tool_search,
    )
    from headroom.proxy.output_effort_policy import can_create_openai_text_verbosity
    from headroom.transforms.thinking_compactor import bills_prior_thinking

    for model, expected in GATE_TABLE:
        if not isinstance(model, str):
            continue
        claude = expected and model_supports_gated_features(model, family="claude")
        gpt = expected and model_supports_gated_features(model, family="gpt")
        assert bills_prior_thinking(model) is bool(claude), model
        assert _model_supports_anthropic_tool_search(model) is bool(claude), model
        assert _model_supports_openai_tool_search(model) is bool(gpt), model
        assert can_create_openai_text_verbosity(model) is bool(gpt), model
