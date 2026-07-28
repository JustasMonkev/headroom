"""Regression tests for the offline / "local-first" safety boundary (issue #11).

Headroom's product language used to read as a promise that requests never leave
the machine ("your data never leaves your machine", "all outbound egress
disabled"). The implementation only consults ``HEADROOM_OFFLINE`` for
*auxiliary* traffic; provider requests still go to the configured upstream.

These tests pin the documented guarantee to the code so the two cannot silently
drift apart:

* ``is_offline()`` gates every auxiliary-egress call site (telemetry, update
  check, license reporting, model downloads).
* Provider forwarding is deliberately **not** gated.
* README / docs / startup log / module comments say exactly that.

If the guarantee is ever changed — e.g. offline mode grows a real provider-egress
block — this file must be updated in the same commit as the docs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from headroom.offline import apply_offline_env, is_offline

REPO_ROOT = Path(__file__).resolve().parents[1]

# Auxiliary-egress call sites that honour HEADROOM_OFFLINE today.
AUXILIARY_EGRESS_MODULES = (
    "headroom/telemetry/beacon.py",  # telemetry beacon
    "headroom/update_check.py",  # release / update check
    "headroom/proxy/server.py",  # license / usage reporter + HF env
    "headroom/offline.py",  # apply_offline_env -> HF_HUB_OFFLINE
    "headroom/subscription/tracker.py",  # Anthropic subscription polling
    "headroom/subscription/copilot_quota.py",  # GitHub Copilot quota polling
    "headroom/subscription/codex_rate_limits.py",  # ChatGPT/Codex usage polling
    "headroom/tokenizers/tiktoken_counter.py",  # BPE vocabulary downloads
)

# Provider forwarding paths. These are NOT gated by HEADROOM_OFFLINE: a request
# to Anthropic/OpenAI/Gemini/Bedrock still leaves the machine in offline mode.
PROVIDER_FORWARDING_MODULES = (
    "headroom/proxy/handlers/anthropic.py",
    "headroom/proxy/handlers/openai.py",
    "headroom/proxy/handlers/gemini.py",
    "headroom/proxy/handlers/bedrock.py",
    "headroom/proxy/handlers/batch.py",
    "headroom/proxy/handlers/streaming.py",
    "headroom/proxy/passthrough.py",
    "headroom/proxy/body_forwarding.py",
)


def _read(rel_path: str) -> str:
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"expected {rel_path} to exist"
    return path.read_text(encoding="utf-8")


# ───────────────────── guarantee 1: auxiliary egress IS gated ─────────────────


class TestAuxiliaryEgressIsGated:
    """Everything the docs claim offline mode disables must actually be gated."""

    def test_telemetry_beacon_is_gated(self, monkeypatch):
        from headroom.telemetry.beacon import is_telemetry_enabled

        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        assert is_telemetry_enabled() is True

        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        assert is_telemetry_enabled() is False

    def test_update_check_is_gated(self, monkeypatch):
        from headroom.update_check import is_update_check_enabled

        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
        monkeypatch.delenv("HEADROOM_UPDATE_CHECK", raising=False)
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        assert is_update_check_enabled() is True

        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        assert is_update_check_enabled() is False

    def test_model_downloads_are_gated(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        apply_offline_env()

        import os

        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_license_reporter_is_gated(self):
        """The UsageReporter phone-home is constructed only when not offline.

        Asserted against the source because building a ``HeadroomProxy`` pulls in
        the whole compression stack; the guard is a single, stable expression.
        """
        source = _read("headroom/proxy/server.py")
        assert re.search(
            r"if\s+config\.license_key\s+and\s+not\s+\(\s*config\.offline\s+or\s+is_offline\(\)\s*\)",
            source,
        ), "UsageReporter construction must stay gated by config.offline / is_offline()"

    def test_subscription_tracking_startup_is_gated(self):
        source = _read("headroom/proxy/server.py")
        assert re.search(
            r"subscription_tracking_active\s*=.*subscription_tracking_enabled\s+and\s+not\s+\("
            r"\s*self\.config\.offline\s+or\s+is_offline\(\)\s*\)",
            source,
            re.DOTALL,
        ), "Subscription tracker startup must stay gated by offline mode"

    @pytest.mark.parametrize("rel_path", AUXILIARY_EGRESS_MODULES)
    def test_auxiliary_module_references_the_switch(self, rel_path):
        source = _read(rel_path)
        assert "is_offline" in source, (
            f"{rel_path} is documented as honouring HEADROOM_OFFLINE but no longer "
            "references is_offline()"
        )


# ─────────────── guarantee 2: provider egress is NOT gated (by design) ────────


class TestProviderEgressIsNotGated:
    """Offline mode is not an egress firewall — the docs must not imply it is.

    This is the deliberate half of the boundary. If a future change *does* gate
    provider forwarding on ``HEADROOM_OFFLINE``, this test fails on purpose: the
    README, ``docs/content/docs/proxy.mdx``, the startup log, and
    ``headroom/offline.py`` all have to be rewritten in the same commit.
    """

    @pytest.mark.parametrize("rel_path", PROVIDER_FORWARDING_MODULES)
    def test_forwarding_path_does_not_consult_offline(self, rel_path):
        source = _read(rel_path)
        assert "is_offline" not in source and "HEADROOM_OFFLINE" not in source, (
            f"{rel_path} now consults the offline switch. Provider egress is "
            "documented as NOT blocked by HEADROOM_OFFLINE — update README.md, "
            "docs/content/docs/proxy.mdx, headroom/offline.py, the proxy startup "
            "log, and this test together."
        )

    def test_offline_mode_does_not_touch_provider_env(self, monkeypatch):
        """apply_offline_env() must not redirect or disable upstream endpoints."""
        import os

        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        apply_offline_env()

        assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
        assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"

    def test_is_offline_is_a_pure_env_predicate(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        assert is_offline() is False
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        assert is_offline() is True
        monkeypatch.setenv("HEADROOM_OFFLINE", "off")
        assert is_offline() is False


# ──────────────── guarantee 3: the docs describe the real boundary ───────────


class TestDocsMatchBehavior:
    def test_readme_does_not_overpromise(self):
        readme = _read("README.md")
        assert "your data never leaves your machine" not in readme.lower(), (
            "README must not claim data never leaves the machine — provider "
            "requests still go to the configured upstream"
        )

    def test_readme_states_the_real_boundary(self):
        readme = _read("README.md").lower()
        assert "never sends your prompts to a" in readme and "headroom-hosted service" in readme
        assert "your prompts still go to your model provider" in readme
        assert "auxiliary" in readme

    def test_proxy_docs_scope_offline_to_auxiliary_egress(self):
        docs = _read("docs/content/docs/proxy.mdx")
        offline_rows = [line for line in docs.splitlines() if "HEADROOM_OFFLINE" in line]
        assert offline_rows, "expected HEADROOM_OFFLINE to be documented in proxy.mdx"
        for row in offline_rows:
            assert "all" not in row.lower().split("egress")[0][-40:], (
                f"proxy.mdx still claims offline disables *all* egress: {row!r}"
            )
        assert "auxiliary" in docs.lower()
        assert "not an egress firewall" in docs.lower()

    def test_startup_log_matches_behavior(self):
        source = _read("headroom/proxy/server.py")
        assert "all outbound egress disabled" not in source, (
            "the proxy startup log must not claim all outbound egress is disabled"
        )
        assert "event=proxy_offline_mode auxiliary egress disabled" in source
        assert "provider requests are NOT blocked" in source

    def test_cli_banner_matches_behavior(self):
        source = _read("headroom/cli/proxy.py")
        assert "OFFLINE (all egress disabled)" not in source
        assert "auxiliary egress disabled; provider traffic unaffected" in source

    def test_offline_module_documents_the_carve_out(self):
        source = _read("headroom/offline.py")
        assert "does not block provider egress" in source.lower()

    def test_offline_module_inventory_includes_all_quota_pollers(self):
        source = _read("headroom/offline.py")
        assert "Eight auxiliary-egress categories" in source
        for module in (
            "headroom.subscription.tracker",
            "headroom.subscription.copilot_quota",
            "headroom.subscription.codex_rate_limits.maybe_schedule_usage_poll",
            "headroom.tokenizers.tiktoken_counter",
        ):
            assert module in source
