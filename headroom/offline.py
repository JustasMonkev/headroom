"""Auxiliary-egress master switch (``HEADROOM_OFFLINE``).

A single predicate the *auxiliary* egress paths consult so a regulated or
air-gapped deployment can disable Headroom's own outbound network access with
one flag. Seven auxiliary-egress categories honour it today:

* ``headroom.telemetry.beacon.is_telemetry_enabled`` — telemetry beacon
* ``headroom.update_check.is_update_check_enabled`` — release/update check
* ``headroom.proxy.server`` (``UsageReporter`` construction) — license /
  usage phone-home
* ``headroom.subscription.tracker`` — Anthropic subscription polling
* ``headroom.subscription.copilot_quota`` — GitHub Copilot quota polling
* ``headroom.subscription.codex_rate_limits.maybe_schedule_usage_poll`` —
  ChatGPT/Codex usage polling
* ``apply_offline_env`` below — HuggingFace / Transformers model downloads

This master switch overrides any path-specific eligibility or opt-out controls,
turns every category off together, and fails closed.

**What this switch does NOT do.** It does not block provider egress. The proxy
forwards requests to whatever upstream is configured (Anthropic, OpenAI, Qwen,
a gateway, or a local endpoint), and that forwarding path never consults
``is_offline()``. Setting ``HEADROOM_OFFLINE=1`` while pointing Headroom at a
remote provider still sends prompts to that provider. A genuine air gap
therefore requires *both* this flag and an upstream that is itself local (e.g.
Ollama / vLLM / LM Studio via ``--api-base``), or a network-level egress block.
Headroom itself never sends prompts to a Headroom-hosted service.

If this ever changes to enforce a provider-egress block, update this docstring,
the startup log in ``headroom/proxy/server.py``, the CLI banner in
``headroom/cli/proxy.py``, the README, ``docs/content/docs/proxy.mdx``, and
``tests/test_offline_guarantees.py`` together — that test pins the wording of
the guarantee to the code.

Kept at the top level (depends only on the stdlib) so any layer — telemetry,
proxy, model code — can import it without creating a package cycle.
"""

from __future__ import annotations

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

OFFLINE_ENV = "HEADROOM_OFFLINE"


def is_offline() -> bool:
    """Return True when ``HEADROOM_OFFLINE`` disables Headroom's auxiliary egress.

    Gates telemetry, update/license checks, quota/subscription polling, and
    model downloads. It does **not** gate provider forwarding — see the module
    docstring.
    """
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in _TRUE_VALUES


def apply_offline_env() -> None:
    """Force HuggingFace/Transformers offline so model code uses only locally
    cached artifacts and never reaches the Hub.

    Idempotent and uses ``setdefault`` so an explicit operator override (e.g.
    ``HF_HUB_OFFLINE=0``) still wins. Call once early in startup.
    """
    if is_offline():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
