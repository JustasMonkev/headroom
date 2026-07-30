"""Server-side Tool Search deferral for OpenAI Responses (gpt-5.4+).

The OpenAI-side analogue of the Anthropic path (issue #746): mark non-core
function / MCP tools ``defer_loading: true`` and inject ``{"type": "tool_search"}``
so OpenAI keeps their heavy parameter schemas out of the model's context until
searched. Gated specifically at GPT 5.4; older models 400 on the fields, and
they keep working through the proxy with ordinary compression instead.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import pytest

from headroom.config import model_supports_gated_features
from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.helpers import (
    _model_supports_openai_tool_search,
    inject_tool_search_deferral_openai,
    is_first_party_openai_target,
    openai_tool_search_enabled,
)


@pytest.fixture(autouse=True)
def _clean_tool_search_env(monkeypatch):
    """The gates read process env; an ambient value would mask the assertions."""
    monkeypatch.delenv("HEADROOM_TOOL_SEARCH", raising=False)
    monkeypatch.delenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", raising=False)


def _fn(name: str) -> dict:
    return {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}


_CORE = ["bash", "read", "write", "edit", "grep", "glob"]
_NONCORE = [f"slack_{i}" for i in range(10)]  # 6 core + 10 non-core = 16 tools (>= min 12)


def _tools() -> list[dict]:
    return [_fn(n) for n in _CORE + _NONCORE]


# --- model gating ------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.4",
        "gpt-5.4-codex",
        "gpt-5.4-2026-02-01",
        "gpt-5.5",
        "gpt-6",
        "gpt-6.2",
        "openai/gpt-5.4",
    ],
)
def test_model_supported(model):
    assert _model_supports_openai_tool_search(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "gpt-5.3",
        "o3",
        "",
        None,
        "claude-opus-4-8",
    ],
)
def test_model_unsupported(model):
    assert _model_supports_openai_tool_search(model) is False


def test_env_override_wins_then_falls_back(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", r"^my-model")
    assert _model_supports_openai_tool_search("my-model-v1") is True
    assert _model_supports_openai_tool_search("gpt-5.5") is False  # override replaces the gate
    # a malformed regex must not crash — fall back to the version gate.
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", "[unclosed")
    assert _model_supports_openai_tool_search("gpt-5.5") is True
    assert _model_supports_openai_tool_search("gpt-5.4") is True
    assert _model_supports_openai_tool_search("gpt-5.3") is False


def test_tool_search_cutoff_does_not_weaken_the_shared_gpt_gate():
    assert _model_supports_openai_tool_search("gpt-5.4") is True
    assert model_supports_gated_features("gpt-5.4", family="gpt") is False


# --- deferral behavior -------------------------------------------------------


def test_defers_non_core_and_injects_search_tool():
    tools = _tools()
    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")
    assert out is not tools  # new list
    assert out[0] == {"type": "tool_search"}  # search tool injected, first, once
    assert sum(1 for t in out if t.get("type") == "tool_search") == 1
    by_name = {t["name"]: t for t in out if t.get("type") == "function"}
    for c in _CORE:
        assert not by_name[c].get("defer_loading")  # core stays resident
    for n in _NONCORE:
        assert by_name[n].get("defer_loading") is True  # non-core deferred


def test_terminal_reserved_namespace_stays_resident():
    terminal = _fn("terminal")
    tools = [terminal] + [_fn(f"peer_{i}") for i in range(11)]
    snapshot = copy.deepcopy(tools)

    out = inject_tool_search_deferral_openai(tools, "gpt-5.6-terra")

    forwarded = next(t for t in out if t.get("name") == "terminal")
    assert forwarded == terminal
    assert "defer_loading" not in forwarded
    assert next(t for t in out if t.get("name") == "peer_0").get("defer_loading") is True
    assert tools == snapshot


def test_memory_search_guidance_stays_resident():
    memory_search = _fn("memory_search")
    tools = [memory_search] + [_fn(f"peer_{i}") for i in range(11)]

    out = inject_tool_search_deferral_openai(tools, "gpt-5.6-terra")

    assert next(t for t in out if t.get("name") == "memory_search") == memory_search
    assert next(t for t in out if t.get("name") == "peer_0").get("defer_loading") is True


def test_terminal_helper_remains_deferrable():
    tools = [_fn("terminal_helper")] + [_fn(f"peer_{i}") for i in range(11)]

    out = inject_tool_search_deferral_openai(tools, "gpt-5.6-terra")

    helper = next(t for t in out if t.get("name") == "terminal_helper")
    assert helper.get("defer_loading") is True


def test_defers_mcp_server():
    tools = [_fn(n) for n in _CORE] + [{"type": "mcp", "server_label": "sentry"}]
    tools += [_fn(f"x{i}") for i in range(8)]
    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")
    mcp = next(t for t in out if t.get("type") == "mcp")
    assert mcp.get("defer_loading") is True


def test_hosted_tools_stay_resident():
    tools = [_fn(n) for n in _CORE] + [{"type": "web_search"}, {"type": "code_interpreter"}]
    tools += [_fn(f"x{i}") for i in range(8)]
    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")
    ws = next(t for t in out if t.get("type") == "web_search")
    ci = next(t for t in out if t.get("type") == "code_interpreter")
    assert "defer_loading" not in ws  # hosted tools can't be deferred
    assert "defer_loading" not in ci


def test_does_not_mutate_input():
    tools = _tools()
    snapshot = copy.deepcopy(tools)
    inject_tool_search_deferral_openai(tools, "gpt-5.5")
    assert tools == snapshot  # deferred tools are copies; the input is untouched


# --- no-op guards ------------------------------------------------------------


def test_noop_for_unsupported_model():
    tools = _tools()
    assert inject_tool_search_deferral_openai(tools, "gpt-4o") is tools


def test_noop_below_min_tools():
    tools = [_fn(f"x{i}") for i in range(5)]  # < 12
    assert inject_tool_search_deferral_openai(tools, "gpt-5.5") is tools


def test_noop_when_tool_search_already_present():
    tools = [{"type": "tool_search"}] + [_fn(f"x{i}") for i in range(15)]
    assert inject_tool_search_deferral_openai(tools, "gpt-5.5") is tools


def test_noop_when_nothing_deferrable():
    tools = [_fn(n) for n in _CORE * 3]  # 18 core tools, none deferrable
    assert inject_tool_search_deferral_openai(tools, "gpt-5.5") is tools


def test_noop_for_non_list():
    assert inject_tool_search_deferral_openai(None, "gpt-5.5") is None


# --- upstream-target gating ---------------------------------------------------
#
# Codex review (helpers.py:3185): `/v1/responses` routed through
# `x-headroom-base-url` to an OpenAI-compatible gateway with a vendor-prefixed
# model such as `openai/gpt-5.5` used to be rejected by the old anchored
# `gpt-…` regex and is now ACCEPTED by the shared parser. Combined with
# deferral being default-on, that injects `tool_search` / `defer_loading` into
# gateways that may not implement them. Automatic injection now requires a
# verified first-party OpenAI target, exactly like the Anthropic path.


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com",
        "https://api.openai.com/v1",
        "http://api.openai.com/v1/responses",
        "api.openai.com",  # bare host, no scheme
        "https://eu.api.openai.com/v1",
        "https://chatgpt.com/backend-api/codex",  # codex login (ChatGPT session auth)
        "wss://chatgpt.com/backend-api/codex/responses",  # the WS Responses upstream
        "https://API.OpenAI.COM/v1",  # case-insensitive
    ],
)
def test_first_party_openai_targets(url):
    assert is_first_party_openai_target(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://gateway.example.com/v1",
        "https://openrouter.ai/api/v1",
        "https://api.githubcopilot.com",  # Copilot speaks Responses, not tool_search
        "https://notopenai.com/v1",
        "https://api.openai.com.evil.example/v1",  # suffix-confusion
        "https://openai.com.evil.example",
        "http://localhost:4000/v1",  # LiteLLM
        "",
        "   ",
        None,
        123,
    ],
)
def test_non_first_party_openai_targets(url):
    assert is_first_party_openai_target(url) is False


def test_auto_injection_requires_a_first_party_target():
    """The regression: a vendor-prefixed id on a custom gateway must NOT inject."""
    assert openai_tool_search_enabled("openai/gpt-5.5", first_party_target=True) is True
    assert openai_tool_search_enabled("openai/gpt-5.5", first_party_target=False) is False
    assert openai_tool_search_enabled("gpt-5.5", first_party_target=False) is False


def test_injection_noops_on_a_custom_gateway_but_fires_first_party():
    tools = _tools()
    assert inject_tool_search_deferral_openai(
        tools, "openai/gpt-5.5", first_party_target=False
    ) is (tools)
    out = inject_tool_search_deferral_openai(tools, "openai/gpt-5.5", first_party_target=True)
    assert out is not tools
    assert out[0] == {"type": "tool_search"}


def test_explicit_env_override_opts_a_compatible_gateway_back_in(monkeypatch):
    """HEADROOM_TOOL_SEARCH=1 is the operator asserting their upstream supports it."""
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "1")
    assert openai_tool_search_enabled("openai/gpt-5.5", first_party_target=False) is True
    out = inject_tool_search_deferral_openai(_tools(), "openai/gpt-5.5", first_party_target=False)
    assert out[0] == {"type": "tool_search"}


def test_model_regex_override_also_opts_a_gateway_in(monkeypatch):
    """The per-deployment model allowlist is the other explicit assertion."""
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", r"^my-gateway-model")
    assert openai_tool_search_enabled("my-gateway-model-1", first_party_target=False) is True
    # ...and it stays a MODEL gate: an id it does not match is still refused.
    assert openai_tool_search_enabled("some-other-model", first_party_target=False) is False


def test_force_on_never_widens_the_model_gate(monkeypatch):
    """`tool_search`/`defer_loading` are hard 400s on pre-5.4 Responses models.

    HEADROOM_TOOL_SEARCH=1 asserts *upstream* support, not model support — the
    model escape hatch is HEADROOM_OPENAI_TOOL_SEARCH_MODELS. Savings profiles
    set HEADROOM_TOOL_SEARCH=1 wholesale, so widening the model gate here would
    start 400ing gpt-4o traffic that works today.
    """
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "1")
    assert openai_tool_search_enabled("gpt-4o", first_party_target=True) is False


def test_kill_switch_disables_the_openai_path_too(monkeypatch):
    """HEADROOM_TOOL_SEARCH=0 is one switch for both providers.

    Savings profiles emit it (`agent_savings.proxy_env`); before this it only
    silenced the Anthropic path while OpenAI kept injecting.
    """
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "0")
    assert openai_tool_search_enabled("gpt-5.5", first_party_target=True) is False
    tools = _tools()
    assert inject_tool_search_deferral_openai(tools, "gpt-5.5", first_party_target=True) is tools


def test_default_first_party_target_preserves_existing_callers():
    """`first_party_target` defaults to True so the helper stays usable standalone."""
    out = inject_tool_search_deferral_openai(_tools(), "gpt-5.5")
    assert out[0] == {"type": "tool_search"}


# --- handler wiring: the gate must see the REAL upstream ----------------------


class _StubTokenizer:
    def count_text(self, text: str) -> int:
        return len(text.split())


class _StubProvider:
    def get_token_counter(self, model: str) -> _StubTokenizer:
        del model
        return _StubTokenizer()


class _StubPipeline:
    transforms: list = []


class _Harness(OpenAIHandlerMixin):
    """Minimal Responses-compression harness pinned to a first-party default."""

    OPENAI_API_URL = "https://api.openai.com"

    def __init__(self) -> None:
        self.openai_pipeline: Any = _StubPipeline()
        self.openai_provider: Any = _StubProvider()


_TS_TRANSFORM = "openai:responses:tool_search_deferral"


def _payload() -> dict:
    return {
        "type": "response.create",
        "model": "openai/gpt-5.4",
        "tools": _tools(),
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
    }


def _transforms_for(upstream: str | None) -> list[str]:
    kwargs = {} if upstream is None else {"upstream_base_url": upstream}
    _working, _mod, _saved, transforms, *_rest = _Harness()._compress_openai_responses_payload(
        _payload(), model="openai/gpt-5.4", request_id="hr_ts_gate", **kwargs
    )
    return list(transforms)


def test_handler_injects_for_a_first_party_openai_upstream():
    assert _TS_TRANSFORM in _transforms_for("https://api.openai.com/v1/responses")


def test_handler_injects_when_no_override_is_given():
    """No `x-headroom-base-url` → the configured OPENAI_API_URL target."""
    assert _TS_TRANSFORM in _transforms_for(None)


def test_handler_skips_injection_for_a_custom_gateway_upstream():
    """The regression: `x-headroom-base-url` → gateway + `openai/gpt-5.4`.

    The shared model parser accepts the vendor-prefixed id the old anchored
    regex rejected, so without the target gate the handler injects
    `tool_search` / `defer_loading` into a gateway that may reject them.
    """
    assert _TS_TRANSFORM not in _transforms_for("https://gateway.example.com/v1")


def test_handler_override_re_enables_a_custom_gateway(monkeypatch):
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "1")
    assert _TS_TRANSFORM in _transforms_for("https://gateway.example.com/v1")


def test_handler_threads_the_upstream_through_the_executor_wrapper():
    """`_compress_..._in_executor` must forward the kwarg, or the gate is blind."""
    sig = inspect.signature(OpenAIHandlerMixin._compress_openai_responses_payload_in_executor)
    assert "upstream_base_url" in sig.parameters
    src = inspect.getsource(OpenAIHandlerMixin.handle_openai_responses)
    assert "upstream_base_url=url" in src, "/v1/responses does not pass its resolved upstream"


def test_malformed_model_override_does_not_authorize_a_gateway(monkeypatch):
    """A bad regex is a typo, not an operator assertion about any upstream.

    `_model_supports_openai_tool_search` swallows `re.error` and falls back to
    the version gate, so a malformed pattern used to leave a nonempty env value
    behind that the custom-upstream check read as authorization — handing
    `tool_search`/`defer_loading` to every gateway on a single bad character.
    """
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", "gpt-5.5[")
    assert openai_tool_search_enabled("gpt-5.5", first_party_target=False) is False
    # A valid, matching override still opts the gateway in.
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", r"gpt-5\.5")
    assert openai_tool_search_enabled("gpt-5.5", first_party_target=False) is True


def test_malformed_model_override_still_falls_back_to_the_version_gate(monkeypatch):
    """Falling back is the documented behaviour and must not regress.

    On a first-party target the malformed pattern is ignored and the version
    gate decides — admitting 5.4+ and still refusing older models.
    """
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", "gpt-5.5[")
    assert openai_tool_search_enabled("gpt-5.5", first_party_target=True) is True
    assert openai_tool_search_enabled("gpt-4o", first_party_target=True) is False
    assert _model_supports_openai_tool_search("gpt-5.5") is True
    assert _model_supports_openai_tool_search("gpt-5.4") is True
    assert _model_supports_openai_tool_search("gpt-5.3") is False
