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

import asyncio
import threading
from contextvars import Context
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


def test_inherited_scope_is_not_mutated_by_concurrent_workers(monkeypatch) -> None:
    """Regression: a scope INHERITED from the caller's context must be copied.

    Context propagation (``copy_context()`` — and therefore
    ``asyncio.to_thread``) is a SHALLOW copy: the two worker contexts hold the
    same outer ``dict`` object. When the calling context already has a routing
    scope open — the sequence below: a direct ``apply()`` on the event-loop
    context, then two concurrent ``asyncio.to_thread(router.apply, ...)`` calls
    — an in-place ``scope[id(router)] = ...`` in the second worker replaces the
    first worker's still-in-use entry, and the first request finishes reading
    the second request's target ratio and tool-args map.

    The existing thread-based tests do NOT cover this: they start from threads
    with no scope at all, so each one takes the ``scope is None`` branch and
    allocates its own map regardless.
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
        tag = getattr(tags, "name", None)
        if tag is not None:
            both_assigned.wait()
            observed[tag] = {
                "target_ratio": self._runtime_target_ratio,
                "tool_call_args": dict(self._tool_call_args),
            }
        return mapping

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
        finally:
            tags.name = None

    async def main() -> None:
        from headroom.transforms.content_router import _open_router_request_scope

        # Plant an inherited caller scope explicitly. Normal apply() calls now
        # reset their scope on exit.
        _open_router_request_scope(router)
        router._runtime_target_ratio = 0.5
        # 2. …which both to_thread workers now inherit, shallow-copied.
        monkeypatch.setattr(ContentRouter, "_build_tool_name_map", probing_build)
        await asyncio.gather(
            asyncio.to_thread(run, "a", 0.2, "toolu_a", "alpha_pattern"),
            asyncio.to_thread(run, "b", 0.9, "toolu_b", "bravo_pattern"),
        )

    asyncio.run(main())

    assert not errors, errors
    assert set(observed) == {"a", "b"}
    assert observed["a"]["target_ratio"] == 0.2
    assert observed["b"]["target_ratio"] == 0.9
    assert "alpha_pattern" in str(observed["a"]["tool_call_args"])
    assert "bravo_pattern" not in str(observed["a"]["tool_call_args"])
    assert "bravo_pattern" in str(observed["b"]["tool_call_args"])
    assert "alpha_pattern" not in str(observed["b"]["tool_call_args"])


def test_opening_a_scope_does_not_disturb_other_routers_in_the_caller(monkeypatch) -> None:
    """Copying the outer map must PRESERVE other routers' live slots.

    A pipeline holding two routers opens a scope per router on the same
    context; the second opening must not lose the first's entry.
    """
    from headroom.transforms.content_router import _open_router_request_scope

    first = ContentRouter(ContentRouterConfig())
    second = ContentRouter(ContentRouterConfig())

    _open_router_request_scope(first)
    first._runtime_target_ratio = 0.33
    _open_router_request_scope(second)
    second._runtime_target_ratio = 0.77

    assert first._runtime_target_ratio == 0.33
    assert second._runtime_target_ratio == 0.77


def test_apply_resets_scope_and_keeps_only_small_diagnostics() -> None:
    """Executor threads must not retain full tool arguments after apply()."""
    from headroom.transforms.content_router import _REQUEST_SCOPE

    router = ContentRouter(ContentRouterConfig())

    def run() -> None:
        router.apply(
            _conversation("toolu_1", "needle"),
            _tokenizer(),
            model_limit=100_000,
            target_ratio=0.42,
        )
        assert _REQUEST_SCOPE.get(None) is None
        assert router._runtime_target_ratio == 0.42
        assert router._tool_call_args == {}
        assert router._tool_call_commands == {}

    Context().run(run)


def test_apply_resets_scope_after_an_error(monkeypatch) -> None:
    from headroom.transforms.content_router import _REQUEST_SCOPE

    router = ContentRouter(ContentRouterConfig())

    def fail(_messages: Any) -> Any:
        raise RuntimeError("routing failed")

    monkeypatch.setattr(router, "_build_tool_name_map", fail)

    def run() -> None:
        try:
            router.apply(_conversation("toolu_1", "needle"), _tokenizer())
        except RuntimeError:
            pass
        else:  # pragma: no cover - the monkeypatch must fail
            raise AssertionError("expected routing failure")
        assert _REQUEST_SCOPE.get(None) is None
        assert router._tool_call_args == {}
        assert router._tool_call_commands == {}

    Context().run(run)


def test_a_recycled_id_does_not_inherit_a_dead_routers_store() -> None:
    """The descriptor's weakref identity guard: a scope entry whose owner has
    been collected must never be handed to a new object that reused its
    ``id()``. Copying the outer map on every scope opening prunes dead entries,
    but a router created *without* opening a scope still meets the stale entry,
    so the guard in ``_RequestScoped._store`` remains load-bearing."""
    from headroom.transforms.content_router import (
        _REQUEST_SCOPE,
        _open_router_request_scope,
    )

    doomed = ContentRouter(ContentRouterConfig())
    _open_router_request_scope(doomed)
    doomed._runtime_target_ratio = 0.123
    doomed_id = id(doomed)
    del doomed

    scope = _REQUEST_SCOPE.get(None)
    assert scope is not None
    # Simulate id() reuse: a fresh router forced onto the dead router's slot.
    fresh = ContentRouter(ContentRouterConfig())
    if doomed_id not in scope:  # already pruned; re-plant to exercise the guard
        import weakref

        collected: ContentRouter | None = ContentRouter(ContentRouterConfig())
        ref = weakref.ref(collected)
        collected = None
        scope[id(fresh)] = (ref, {"_runtime_target_ratio": 0.123})
    else:
        scope[id(fresh)] = scope.pop(doomed_id)

    # The stale store says 0.123; the identity check must reject it.
    assert fresh._runtime_target_ratio is None
    fresh._runtime_target_ratio = 0.456
    assert fresh._runtime_target_ratio == 0.456


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
