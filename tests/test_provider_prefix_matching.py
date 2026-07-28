"""Longest-prefix model resolution in the OpenAI provider.

First-match iteration over `_PRICING`/`_CONTEXT_LIMITS`/`_MODEL_ENCODINGS`
was insertion-order-dependent: "gpt-4o-mini-2024-07-18" resolved via the
"gpt-4o" entry (16.7x the real price) and "o1-mini-2024-09-12" via "o1"
(claiming a 200k context for a 128k model, so compression never gated and
the request 400'd upstream). Longest prefix always picks the most specific
table entry.
"""

from __future__ import annotations

import pytest

from headroom.providers.openai import OpenAIProvider, _get_encoding_name_for_model


@pytest.fixture()
def provider() -> OpenAIProvider:
    return OpenAIProvider()


class TestPricingPrefix:
    def test_dated_mini_snapshot_prices_as_mini(self, provider: OpenAIProvider) -> None:
        assert provider._get_pricing("gpt-4o-mini-2024-07-18") == provider._get_pricing(
            "gpt-4o-mini"
        )
        assert provider._get_pricing("gpt-4o-mini-2024-07-18") != provider._get_pricing("gpt-4o")

    def test_dated_o_series_mini_prices_as_mini(self, provider: OpenAIProvider) -> None:
        assert provider._get_pricing("o1-mini-2024-09-12") == provider._get_pricing("o1-mini")
        assert provider._get_pricing("o3-mini-2025-01-31") == provider._get_pricing("o3-mini")


class TestContextLimitPrefix:
    def test_o1_mini_snapshot_gets_mini_context(self, provider: OpenAIProvider) -> None:
        assert provider._get_context_limit_manual(
            "o1-mini-2024-09-12"
        ) == provider._get_context_limit_manual("o1-mini")

    def test_gpt4_32k_snapshot_gets_32k_context(self, provider: OpenAIProvider) -> None:
        assert provider._get_context_limit_manual("gpt-4-32k-0613") == 32768


class TestCustomOverridePrecedence:
    def test_family_override_beats_longer_builtin_prefix(self) -> None:
        """A caller-supplied family override like {"o1": 64000} must apply to
        dated snapshots even though the built-in "o1-mini" key is a longer
        prefix — the constructor documents that explicit overrides take
        precedence."""
        provider = OpenAIProvider(context_limits={"o1": 64000})
        assert provider._get_context_limit_manual("o1-mini-2024-09-12") == 64000
        assert provider._get_context_limit_manual("o1-2024-12-17") == 64000
        # Unrelated families still resolve from the built-in table.
        assert provider._get_context_limit_manual("gpt-4-32k-0613") == 32768

    def test_more_specific_custom_override_wins_within_custom(self) -> None:
        provider = OpenAIProvider(context_limits={"o1": 64000, "o1-mini": 32000})
        assert provider._get_context_limit_manual("o1-mini-2024-09-12") == 32000
        assert provider._get_context_limit_manual("o1-2024-12-17") == 64000


class TestEncodingPrefix:
    def test_known_families_resolve(self) -> None:
        assert _get_encoding_name_for_model("gpt-4o-2024-11-20") == "o200k_base"
        assert _get_encoding_name_for_model("gpt-4-32k-0613") == "cl100k_base"

    def test_custom_encodings_still_take_priority(self) -> None:
        assert (
            _get_encoding_name_for_model("gpt-4o", custom_encodings={"gpt-4o": "cl100k_base"})
            == "cl100k_base"
        )


class TestCustomLayerPrecedence:
    def test_explicit_override_beats_longer_config_file_key(self, monkeypatch) -> None:
        """A longer prefix in a LOWER-priority source (config file) must not
        out-rank a shorter explicit constructor override."""
        import headroom.providers.openai as mod

        monkeypatch.setattr(
            mod,
            "_load_custom_model_config",
            lambda: {
                "context_limits": {"o1-mini": 128000},
                "pricing": {},
                "encodings": {},
            },
        )
        provider = OpenAIProvider(context_limits={"o1": 64000})
        assert provider._get_context_limit_manual("o1-mini-2024-09-12") == 64000
        # Without an explicit override, the config-file entry applies.
        provider2 = OpenAIProvider()
        assert provider2._get_context_limit_manual("o1-mini-2024-09-12") == 128000
