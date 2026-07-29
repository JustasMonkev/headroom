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

import inspect
from typing import Any

import pytest

from headroom.proxy.handlers import anthropic as anthropic_handler
from headroom.proxy.helpers import (
    _model_supports_anthropic_tool_search,
    anthropic_tool_search_enabled,
    inject_tool_search_deferral,
    is_first_party_anthropic_target,
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


# --------------------------------------------------------------------------
# Upstream identity: automatic injection is for FIRST-PARTY Anthropic only.
#
# `/v1/messages` routed through `x-headroom-base-url` keeps provider_name
# "anthropic" (the CLIENT speaks the Anthropic wire format) while the UPSTREAM
# is an arbitrary compatible gateway. Such a gateway need not implement
# Anthropic Tool Search, so default-on injection of `tool_search_tool_*` /
# `defer_loading` would break requests that previously passed through.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://api.anthropic.com",
        "https://api.anthropic.com/",
        "HTTPS://API.ANTHROPIC.COM",
        "api.anthropic.com",  # scheme-less config value
        "https://eu.anthropic.com/v1",
    ],
)
def test_first_party_anthropic_targets_are_recognized(url: str) -> None:
    assert is_first_party_anthropic_target(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://my-gateway.example.com",
        "https://openrouter.ai/api/v1",
        "http://localhost:8080",
        "https://anthropic.com.evil.example",  # suffix must be a real host boundary
        "https://api.anthropic.com.evil.example",
        "",
        "   ",
        None,
    ],
)
def test_non_first_party_targets_fail_closed(url: Any) -> None:
    assert is_first_party_anthropic_target(url) is False


def test_auto_injection_requires_a_first_party_upstream() -> None:
    """A supported model behind a custom gateway must NOT get auto-injection."""
    assert anthropic_tool_search_enabled(OPUS_48, first_party_target=True) is True
    assert anthropic_tool_search_enabled(OPUS_48, first_party_target=False) is False


def test_explicit_force_on_still_overrides_the_upstream_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HEADROOM_TOOL_SEARCH=1 is the operator asserting their upstream supports it."""
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "1")
    assert anthropic_tool_search_enabled(OPUS_48, first_party_target=False) is True


def test_opt_out_still_wins_over_a_first_party_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "0")
    assert anthropic_tool_search_enabled(OPUS_48, first_party_target=True) is False


def test_handler_feeds_the_per_request_gateway_into_the_gate() -> None:
    """The handler must pass the effective upstream, not assume first-party."""
    src = inspect.getsource(anthropic_handler.AnthropicHandlerMixin.handle_anthropic_messages)
    idx = src.find("anthropic_tool_search_enabled(\n")
    assert idx != -1, "the gate is not called with per-request arguments"
    window = src[idx : idx + 400]
    assert "first_party_target=" in window, "upstream identity not carried into the gate"
    assert "is_first_party_anthropic_target(" in window
    assert "upstream_base_url" in window, "per-request gateway override ignored"


# --------------------------------------------------------------------------
# The deferral must never destroy the tools cache breakpoint it exists to protect.
# --------------------------------------------------------------------------
def test_cache_control_survives_when_every_custom_tool_would_be_deferred() -> None:
    """All-custom tool set + a client breakpoint: the breakpoint must survive.

    With no core tool present, every real tool is deferrable — so there is no
    resident real tool to move `cache_control` onto and the breakpoint used to
    be dropped outright, silently ending prefix caching for the tools block.
    """
    tools: list[Any] = _tools(14)
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

    out = inject_tool_search_deferral(tools)

    assert out is not tools
    breakpoints = [t for t in out if isinstance(t, dict) and t.get("cache_control")]
    assert len(breakpoints) == 1, f"exactly one tools breakpoint must survive: {breakpoints}"
    holder = breakpoints[0]
    assert not holder.get("defer_loading"), "a deferred tool must never carry cache_control"
    # The breakpoint stays on the tool the client put it on (byte-stable prefix),
    # and everything else is still deferred.
    assert holder["name"] == tools[-1]["name"]
    assert sum(1 for t in out if isinstance(t, dict) and t.get("defer_loading")) == 13
    # Input is never mutated in place.
    assert "defer_loading" not in tools[-1]


def test_cache_control_moves_to_a_resident_tool_when_one_exists() -> None:
    """With a core tool resident, the breakpoint moves there and all custom
    tools stay deferred (the pre-existing path, unchanged)."""
    tools: list[Any] = [
        {"name": "Bash", "description": "core", "input_schema": {}},
        *_tools(13),
    ]
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

    out = inject_tool_search_deferral(tools)

    breakpoints = [t for t in out if isinstance(t, dict) and t.get("cache_control")]
    assert len(breakpoints) == 1
    assert breakpoints[0]["name"] == "Bash"
    assert not breakpoints[0].get("defer_loading")
    assert sum(1 for t in out if isinstance(t, dict) and t.get("defer_loading")) == 13


def test_no_breakpoint_means_everything_is_still_deferred() -> None:
    """Without a client breakpoint there is nothing to protect — defer all."""
    tools = _tools(14)
    out = inject_tool_search_deferral(tools)
    assert sum(1 for t in out if isinstance(t, dict) and t.get("defer_loading")) == 14
    assert not any(isinstance(t, dict) and t.get("cache_control") for t in out)
