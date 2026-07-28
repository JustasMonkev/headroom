"""F6: per-request routing state must not bleed across concurrent requests.

`ContentRouter` instances are long-lived and shared (one per proxy pipeline),
`apply()` runs on a compression executor thread, and the block-compression pass
fans out to more threads. Request-specific routing knobs — `_runtime_target_ratio`,
`_runtime_kompress_model`, the `tool_call_id -> args/command` maps and the
read-protection sets — used to be plain instance attributes assigned at the top
of `apply()`, so two in-flight requests overwrote each other's routing context.

These tests interleave two `apply()` calls on ONE router and assert each side
only ever observes its own values.
"""

from __future__ import annotations

import threading
from typing import Any

from headroom.tokenizer import Tokenizer
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


class _CountingTokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.count_text(str(m.get("content", ""))) for m in messages)


def _tokenizer() -> Tokenizer:
    return Tokenizer(_CountingTokenizer(), "test-model")  # type: ignore[arg-type]


def _conversation(call_id: str, pattern: str) -> list[dict[str, Any]]:
    """One completed tool call whose args are distinctive per request."""
    return [
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": "Grep",
                    "input": {"pattern": pattern, "path": "/repo"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": "hit\n" * 40}],
        },
    ]


def test_concurrent_applies_do_not_share_runtime_routing_state(monkeypatch) -> None:
    """Two interleaved requests with different target ratios / tool-args maps.

    A class-level probe on `_build_tool_name_map` (called by `apply()` right
    after the runtime kwargs are installed) parks each thread on a barrier so
    BOTH requests have written their state before either reads it back. With
    the old shared-instance-attribute design the second writer wins and the
    first thread reads the other request's values.
    """
    router = ContentRouter(ContentRouterConfig())
    tokenizer = _tokenizer()

    both_assigned = threading.Barrier(2, timeout=30)
    observed: dict[str, dict[str, Any]] = {}
    errors: list[BaseException] = []
    tags = threading.local()
    original_build = ContentRouter._build_tool_name_map

    def probing_build(self: ContentRouter, messages: Any) -> Any:
        mapping = original_build(self, messages)
        both_assigned.wait()
        observed[tags.name] = {
            "target_ratio": self._runtime_target_ratio,
            "tool_call_args": dict(self._tool_call_args),
        }
        return mapping

    monkeypatch.setattr(ContentRouter, "_build_tool_name_map", probing_build)

    def run(tag: str, ratio: float, call_id: str, pattern: str) -> None:
        tags.name = tag
        try:
            router.apply(
                _conversation(call_id, pattern),
                tokenizer,
                model_limit=100_000,
                target_ratio=ratio,
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
            both_assigned.abort()

    t1 = threading.Thread(target=run, args=("a", 0.2, "toolu_a", "alpha_pattern"))
    t2 = threading.Thread(target=run, args=("b", 0.9, "toolu_b", "bravo_pattern"))
    t1.start()
    t2.start()
    t1.join(60)
    t2.join(60)

    assert not errors, errors
    assert set(observed) == {"a", "b"}
    assert observed["a"]["target_ratio"] == 0.2
    assert observed["b"]["target_ratio"] == 0.9
    assert "alpha_pattern" in str(observed["a"]["tool_call_args"])
    assert "bravo_pattern" not in str(observed["a"]["tool_call_args"])
    assert "bravo_pattern" in str(observed["b"]["tool_call_args"])
    assert "alpha_pattern" not in str(observed["b"]["tool_call_args"])


def test_scope_is_isolated_between_threads_without_apply() -> None:
    """Direct assignment inside a scope must not leak to another thread."""
    from headroom.transforms.content_router import _open_router_request_scope

    router = ContentRouter(ContentRouterConfig())
    seen: dict[str, Any] = {}
    ready = threading.Barrier(2, timeout=20)

    def worker(tag: str, value: float) -> None:
        _open_router_request_scope(router)
        router._runtime_target_ratio = value
        ready.wait()
        seen[tag] = router._runtime_target_ratio

    t1 = threading.Thread(target=worker, args=("x", 0.11))
    t2 = threading.Thread(target=worker, args=("y", 0.99))
    t1.start()
    t2.start()
    t1.join(30)
    t2.join(30)

    assert seen == {"x": 0.11, "y": 0.99}


def test_state_is_readable_after_apply_returns() -> None:
    """Post-`apply()` inspection (tests, savings reporting) still works."""
    router = ContentRouter(ContentRouterConfig())
    router.apply(
        _conversation("toolu_1", "needle"),
        _tokenizer(),
        model_limit=100_000,
        target_ratio=0.42,
    )
    assert router._runtime_target_ratio == 0.42
    assert "needle" in str(router._tool_call_args)


def test_defaults_are_available_without_any_scope() -> None:
    """A fresh router read outside apply() yields the documented defaults."""
    router = ContentRouter(ContentRouterConfig())
    assert router._runtime_target_ratio is None
    assert router._runtime_kompress_model is None
    assert router._runtime_compression_policy is None
    assert router._tool_call_args == {}
    assert router._tool_call_commands == {}
    assert router._protect_read_tool_ids == set()
    assert router._protect_read_msg_indices == set()
