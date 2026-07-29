"""Pure output-effort policy decisions.

The output shaper mutates provider request bodies. This module owns the
provider-neutral decisions behind those mutations so rank comparisons and
legacy budget clamping stay testable without request dictionaries.
"""

from __future__ import annotations

from headroom.config import model_supports_gated_features

EFFORT_RANK = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4}
TEXT_VERBOSITY_RANK = {"low": 0, "medium": 1, "high": 2}
LEGACY_THINKING_FLOOR = 1024


def lower_effort_value(current: object, target: str) -> str | None:
    """Return ``target`` when an existing effort should be lowered."""
    if not isinstance(current, str):
        return None
    if current not in EFFORT_RANK or target not in EFFORT_RANK:
        return None
    if EFFORT_RANK[current] <= EFFORT_RANK[target]:
        return None
    return target


def clamp_legacy_thinking_budget(
    *,
    thinking_type: object,
    budget_tokens: object,
    floor: int = LEGACY_THINKING_FLOOR,
) -> int | None:
    """Return the clamped budget for legacy enabled thinking, else ``None``."""
    if thinking_type != "enabled":
        return None
    if not isinstance(budget_tokens, int):
        return None
    if budget_tokens <= floor:
        return None
    return floor


def can_create_openai_text_verbosity(
    model: object,
    *,
    first_party_target: bool = False,
) -> bool:
    """Whether it is safe to CREATE a new OpenAI ``text.verbosity`` block.

    Two independent conditions, both required:

    * **Model** — native output controls are a model-specific optimization, so
      they engage only at or above the shared cutoff
      (``MIN_GPT_FEATURE_VERSION``, gpt >= 5.5).
    * **Upstream** — ``text.verbosity`` is an OpenAI-Responses-specific field.
      A ``/v1/responses`` request routed through ``x-headroom-base-url`` still
      *looks* like an OpenAI request (the client speaks the Responses wire
      format) while the upstream is an arbitrary compatible gateway that may
      reject the field. That gap widened when the shared model parser started
      accepting vendor-prefixed ids such as ``openai/gpt-5.5`` — exactly the
      ids gateways use — where the old anchored ``gpt-…`` regex rejected them.
      So ``first_party_target`` must carry the *verified upstream identity*
      (:func:`headroom.proxy.helpers.is_first_party_openai_target` applied to
      the already-resolved destination), never the request dialect.

    Fail-closed on both axes, and ``first_party_target`` defaults to ``False``
    so a call site that never learned the upstream cannot silently create the
    field. This deliberately differs from
    :func:`~headroom.proxy.helpers.openai_tool_search_enabled`, whose flag
    defaults to ``True``: that gate ships explicit operator opt-ins
    (``HEADROOM_TOOL_SEARCH=1``, ``HEADROOM_OPENAI_TOOL_SEARCH_MODELS``) that
    can re-enable a gateway, whereas this one has no escape hatch — so here the
    default *is* the policy, and it must be the safe one.

    Nothing is broken by failing closed: the request keeps flowing, it just
    falls back to the portable instruction-steering lever (which is
    upstream-agnostic), and an *existing* client-sent verbosity is still
    lowered — for every model and every upstream — by
    :func:`lower_text_verbosity_value`. The client having sent the field is
    itself proof that the target accepts it.
    """
    if not first_party_target:
        return False
    return model_supports_gated_features(model, family="gpt")


def lower_text_verbosity_value(current: object) -> str | None:
    """Return ``low`` when an existing OpenAI text verbosity should be lowered."""
    if not isinstance(current, str):
        return None
    if current not in TEXT_VERBOSITY_RANK:
        return None
    if TEXT_VERBOSITY_RANK[current] <= TEXT_VERBOSITY_RANK["low"]:
        return None
    return "low"
