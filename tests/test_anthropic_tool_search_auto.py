"""F1: Anthropic server-side Tool Search deferral is default-ON (`auto`).

The deferral machinery (`inject_tool_search_deferral`) already existed but was
gated behind an opt-in `HEADROOM_TOOL_SEARCH` env var, so the ~28k tokens of
tool schemas an MCP-heavy client ships on every request were never deferred by
default. These tests pin the new resolution: unset/`auto` => on for models that
support the GA tool-search shape, `0`/`off` => explicit opt-out, `1`/`on` =>
force on regardless of the model version gate.

The version gate is the shared model-feature cutoff (`MIN_CLAUDE_FEATURE_VERSION`
= Claude 4.8): model-specific optimizations engage only on recent frontier
models. Older Claude models still proxy fine and still get ordinary
compression — they just don't get tool-search deferral.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.helpers import (
    _model_supports_anthropic_tool_search,
    anthropic_tool_search_enabled,
    inject_tool_search_deferral,
    tool_search_mode,
)

# At/above the shared cutoff (Claude >= 4.8) -> gated features engage.
OPUS_48 = "claude-opus-4-8"
SONNET_5 = "claude-sonnet-5"
# Below the cutoff -> proxied normally, but no tool-search deferral.
SONNET_45 = "claude-sonnet-4-5-20250929"
OPUS_46 = "claude-opus-4-6"
LEGACY = "claude-3-5-sonnet-20241022"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_TOOL_SEARCH", raising=False)
    monkeypatch.delenv("HEADROOM_TOOL_SEARCH_MODELS", raising=False)


# --------------------------------------------------------------------------
# Mode resolution
# --------------------------------------------------------------------------
def test_unset_env_resolves_to_auto() -> None:
    assert tool_search_mode() == "auto"


@pytest.mark.parametrize("value", ["auto", "AUTO", " Auto "])
def test_explicit_auto_stays_auto(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", value)
    assert tool_search_mode() == "auto"


@pytest.mark.parametrize("value", ["0", "off", "false", "no", "OFF", " 0 "])
def test_opt_out_values_resolve_to_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", value)
    assert tool_search_mode() == "off"
    assert anthropic_tool_search_enabled(OPUS_48) is False


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "TRUE"])
def test_explicit_on_values_still_work(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", value)
    assert tool_search_mode() == "on"
    # Force-on bypasses the version gate (escape hatch for unknown model IDs).
    assert anthropic_tool_search_enabled(LEGACY) is True
    assert anthropic_tool_search_enabled("some-future-model") is True


def test_unrecognized_value_falls_back_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "maybe")
    assert tool_search_mode() == "auto"
    assert anthropic_tool_search_enabled(OPUS_48) is True


# --------------------------------------------------------------------------
# Model gating (auto mode)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [OPUS_48, SONNET_5, "claude-sonnet-5-1", "anthropic/claude-opus-4-8"],
)
def test_auto_is_on_for_supported_models(model: str) -> None:
    assert _model_supports_anthropic_tool_search(model) is True
    assert anthropic_tool_search_enabled(model) is True


@pytest.mark.parametrize(
    "model",
    [
        SONNET_45,
        OPUS_46,
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-7",
        LEGACY,
        "claude-3-opus-20240229",
        "claude-sonnet-4",
        "",
        None,
    ],
)
def test_auto_is_off_below_the_feature_cutoff(model: Any) -> None:
    """Below Claude 4.8 the model still works — it just skips this feature."""
    assert _model_supports_anthropic_tool_search(model) is False
    assert anthropic_tool_search_enabled(model) is False


def test_model_regex_override_wins_over_version_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_MODELS", r"^claude-3-5")
    assert _model_supports_anthropic_tool_search(LEGACY) is True
    assert _model_supports_anthropic_tool_search(OPUS_48) is False


def test_malformed_model_regex_falls_back_to_version_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_MODELS", "[unclosed")
    assert _model_supports_anthropic_tool_search(OPUS_48) is True
    assert _model_supports_anthropic_tool_search(OPUS_46) is False
    assert _model_supports_anthropic_tool_search(LEGACY) is False


# --------------------------------------------------------------------------
# The default-on path still respects the injector's own no-op conditions.
# --------------------------------------------------------------------------
def _tools(n: int) -> list[dict[str, Any]]:
    return [
        {"name": f"mcp__slack__op_{i}", "description": "x" * 200, "input_schema": {}}
        for i in range(n)
    ]


def test_default_on_defers_a_large_mcp_toolset() -> None:
    assert anthropic_tool_search_enabled(OPUS_48) is True
    tools = _tools(30)
    out = inject_tool_search_deferral(tools)
    assert out is not tools
    assert out[0]["type"].startswith("tool_search_tool_")
    assert sum(1 for t in out if isinstance(t, dict) and t.get("defer_loading")) == 30


def test_default_on_still_no_ops_below_the_min_tools_threshold() -> None:
    tools = _tools(4)
    assert inject_tool_search_deferral(tools) is tools


def test_default_on_still_no_ops_when_client_already_defers() -> None:
    tools: list[Any] = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        *_tools(20),
    ]
    assert inject_tool_search_deferral(tools) is tools
