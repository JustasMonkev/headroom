"""The compression deadline must bound the WHOLE pass-2 stage (issue #7).

``tests/test_content_router_single_item_deadline.py`` only ever builds a single
pending task, which was the one shape ``HEADROOM_COMPRESSION_DEADLINE_MS`` used
to cover: the watchdog was armed with ``_compression_deadline_seconds() if
len(pending_tasks) == 1 else 0.0``. With two or more cache misses — the normal
case for a real conversation, and the case ``--mode token`` reaches constantly —
the deadline was hard-coded to ``0.0``, the inline loop ran unbounded, and the
thread-pool branch blocked on ``Future.result()`` with no timeout. A single
wedged compressor hung the request forever and the documented mitigation env var
appeared to do nothing.

These tests pin the multi-block shape on both branches.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest

from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
    _DaemonBoundedExecutor,
)


class _Tokenizer:
    def count_text(self, content: str) -> int:
        return len(content.split())


def _compression_result(content: str, compressed: str) -> RouterCompressionResult:
    return RouterCompressionResult(
        compressed=compressed,
        original=content,
        strategy_used=CompressionStrategy.TEXT,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=len(content.split()),
                compressed_tokens=len(compressed.split()),
            )
        ],
    )


def _router() -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            protect_recent_code=0,
            protect_analysis_context=False,
            skip_user_messages=False,
        )
    )


def _messages(pending: int = 3) -> list[dict[str, str]]:
    """One frozen prefix message plus ``pending`` compressible cache misses."""
    out = [{"role": "assistant", "content": "frozen prefix content remains unchanged"}]
    for i in range(pending):
        out.append(
            {
                "role": "assistant",
                "content": f"pending cache miss number {i} with enough words to compress",
            }
        )
    return out


def _apply(router: ContentRouter, messages: list[dict[str, str]]):
    return router.apply(
        messages,
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )


@pytest.fixture(autouse=True)
def _warm_detector():
    """Settle the content detector before any timing assertion.

    Pass 1 runs content detection behind its own watchdog
    (``HEADROOM_DETECT_TIMEOUT_SECS``, default 5s). On first use in a cold
    process the native detector can burn that whole watchdog before the
    circuit breaker disables it — time that has nothing to do with the
    compression stage budget under test here. Warm it once with a trivially
    fast compressor so the timings below measure pass 2 only.
    """
    router = _router()
    router.compress = lambda content, *, context="", bias=1.0: _compression_result(content, "x")
    _apply(router, _messages())


def test_multi_block_parallel_branch_fails_open_at_deadline(monkeypatch, caplog):
    """The default (thread-pool) branch must not block past the stage budget."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "4")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "50")

    release = threading.Event()

    def slow_compress(content, *, context="", bias=1.0):
        release.wait(timeout=5.0)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", slow_compress)

    original = _messages()
    started = time.perf_counter()
    try:
        result = _apply(router, original)
        elapsed = time.perf_counter() - started
    finally:
        release.set()

    # Bounded by the budget, not by the 5s sleep.
    assert elapsed < 2.0, f"stage took {elapsed:.2f}s — deadline did not bound the thread pool"
    # Every block failed open with its content intact.
    for i in range(1, len(original)):
        assert result.messages[i]["content"] == original[i]["content"]
    assert "failing open via PASSTHROUGH" in caplog.text


def test_multi_block_inline_branch_fails_open_at_deadline(monkeypatch, caplog):
    """Same guarantee with parallelism disabled (HEADROOM_COMPRESS_WORKERS=1)."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "1")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "50")

    release = threading.Event()

    def slow_compress(content, *, context="", bias=1.0):
        release.wait(timeout=5.0)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", slow_compress)

    original = _messages()
    started = time.perf_counter()
    try:
        result = _apply(router, original)
        elapsed = time.perf_counter() - started
    finally:
        release.set()

    assert elapsed < 2.0, f"stage took {elapsed:.2f}s — deadline did not bound the inline loop"
    for i in range(1, len(original)):
        assert result.messages[i]["content"] == original[i]["content"]
    assert "failing open via PASSTHROUGH" in caplog.text


def test_stage_budget_is_shared_not_per_task(monkeypatch):
    """A slow first block must not grant every later block a fresh full budget.

    With N blocks each sleeping just under a per-task deadline, a per-task
    budget would let the stage run N x deadline. The budget is per stage.
    """
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "1")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "100")

    def slow_compress(content, *, context="", bias=1.0):
        time.sleep(0.3)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", slow_compress)

    started = time.perf_counter()
    _apply(router, _messages(pending=5))
    elapsed = time.perf_counter() - started

    # 5 blocks x 0.3s = 1.5s if the budget reset per task; one 0.1s budget total.
    assert elapsed < 0.9, f"stage took {elapsed:.2f}s — budget appears to reset per task"


def test_multi_block_under_deadline_still_compresses(monkeypatch):
    """The bound must not cost savings when compression is comfortably fast."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "4")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "5000")
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(content, "compressed output"),
    )

    result = _apply(router, _messages())

    for i in range(1, len(_messages())):
        assert result.messages[i]["content"] == "compressed output"


def test_multi_block_disabled_deadline_still_compresses(monkeypatch):
    """HEADROOM_COMPRESSION_DEADLINE_MS=0 disables the bound, as before."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "4")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "0")
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(content, "compressed output"),
    )

    result = _apply(router, _messages())

    for i in range(1, len(_messages())):
        assert result.messages[i]["content"] == "compressed output"


def test_completed_later_futures_survive_earlier_timeout(monkeypatch):
    """Only unfinished work fails open after the shared deadline expires."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "4")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "50")

    def first_is_slow(content, *, context="", bias=1.0):
        if "number 0" in content:
            time.sleep(0.3)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", first_is_slow)
    original = _messages()
    result = _apply(router, original)

    assert result.messages[1]["content"] == original[1]["content"]
    assert result.messages[2]["content"] == "compressed output"
    assert result.messages[3]["content"] == "compressed output"


def test_timed_out_workers_are_bounded_across_requests(monkeypatch):
    """Repeated deadlines bound both live workers and retained queued work."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "4")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "30")
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def wedged_compress(content, *, context="", bias=1.0):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            release.wait(timeout=2.0)
            return _compression_result(content, "compressed output")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(router, "compress", wedged_compress)
    try:
        # Fill all four workers plus the executor's one bounded pending wave.
        _apply(router, _messages(pending=12))
        executor = router._stage_compression_executor
        assert executor is not None
        original_submit = executor.submit
        later_accepted = 0

        def count_submit(*args, **kwargs):
            nonlocal later_accepted
            future = original_submit(*args, **kwargs)
            if future is not None:
                later_accepted += 1
            return future

        monkeypatch.setattr(executor, "submit", count_submit)
        for _ in range(3):
            _apply(router, _messages(pending=12))

        assert max_active <= 4
        assert router._stage_compression_executor_workers == 4
        assert router._stage_compression_admission_capacity == 8
        assert later_accepted == 0, "saturated executor accepted more retained request work"
        assert all(worker.daemon for worker in executor._threads)
    finally:
        release.set()
        deadline = time.monotonic() + 1.0
        while active and time.monotonic() < deadline:
            time.sleep(0.01)


def test_short_lived_routers_reuse_the_process_worker_pool():
    """Standalone optimize calls must not leave one idle pool per router."""
    first = _router()
    second = _router()

    first_executor = first._get_stage_compression_executor(3)
    second_executor = second._get_stage_compression_executor(3)

    assert first_executor is second_executor
    assert len(first_executor._threads) <= 3


def test_large_worker_setting_starts_only_workers_needed_for_one_block(monkeypatch):
    """A large tuning ceiling must not eagerly allocate every daemon thread."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "64")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "2000")
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(content, "compressed output"),
    )

    result = _apply(router, _messages(pending=1))
    executor = router._stage_compression_executor

    assert result.messages[1]["content"] == "compressed output"
    assert executor is not None
    assert len(executor._threads) == 1


def test_idle_worker_releases_completed_payload_references():
    """An idle worker must not pin the prior request's content in its frame."""

    class _Payload:
        pass

    executor = _DaemonBoundedExecutor(max_workers=1, max_pending=1)
    payload = _Payload()
    payload_ref = weakref.ref(payload)
    future = executor.submit(lambda _payload: None, payload, block=False)
    assert future is not None
    future.result(timeout=1.0)

    del future, payload
    deadline = time.monotonic() + 1.0
    while payload_ref() is not None and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)

    assert payload_ref() is None


def test_lazy_executor_expands_for_concurrent_pending_work():
    """A busy first worker must not leave the second task queued behind it."""
    executor = _DaemonBoundedExecutor(max_workers=2, max_pending=2)
    release = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()

    def block(started: threading.Event):
        started.set()
        release.wait(timeout=2.0)

    first = executor.submit(block, first_started, block=False)
    assert first is not None
    assert first_started.wait(timeout=1.0)
    second = executor.submit(block, second_started, block=False)
    assert second is not None
    try:
        assert second_started.wait(timeout=1.0)
        assert len(executor._threads) == 2
    finally:
        release.set()


def test_burst_larger_than_queue_drains_in_waves(monkeypatch):
    """Healthy work beyond running+queued capacity still runs before deadline."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "2")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "2000")

    def fast_compress(content, *, context="", bias=1.0):
        time.sleep(0.01)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", fast_compress)

    result = _apply(router, _messages(pending=12))

    for message in result.messages[1:]:
        assert message["content"] == "compressed output"


def test_deadline_fallback_is_retried_instead_of_skip_cached(monkeypatch):
    """A timeout says nothing about whether the same content can compress later."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "3")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "30")
    release = threading.Event()

    def wedged_compress(content, *, context="", bias=1.0):
        release.wait(timeout=2.0)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", wedged_compress)
    original = _messages(pending=1)
    try:
        first = _apply(router, original)
    finally:
        release.set()
    assert first.messages[1]["content"] == original[1]["content"]

    # Allow the timed-out worker to leave the shared executor before retrying.
    deadline = time.monotonic() + 1.0
    executor = router._stage_compression_executor
    assert executor is not None
    while executor._queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)

    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(content, "compressed output"),
    )
    second = _apply(router, original)

    assert second.messages[1]["content"] == "compressed output"


def test_deadline_cancels_queued_compression_work(monkeypatch):
    """Queued work must not execute after its request has already failed open."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "2")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "30")
    release = threading.Event()
    lock = threading.Lock()
    started: list[str] = []

    def wedged_compress(content, *, context="", bias=1.0):
        with lock:
            started.append(content)
        release.wait(timeout=2.0)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", wedged_compress)
    try:
        _apply(router, _messages(pending=8))
        with lock:
            assert len(started) == 2
    finally:
        release.set()

    executor = router._stage_compression_executor
    assert executor is not None
    deadline = time.monotonic() + 1.0
    while executor._queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)

    with lock:
        assert len(started) == 2, "cancelled queued compression ran after the deadline"


def test_inline_deadline_cancels_work_waiting_in_shared_queue(monkeypatch):
    """A one-block request must remove its timed-out queued future."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "2")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "30")
    release = threading.Event()
    blockers_started = threading.Event()
    lock = threading.Lock()
    active_blockers = 0

    def blocker():
        nonlocal active_blockers
        with lock:
            active_blockers += 1
            if active_blockers == 2:
                blockers_started.set()
        release.wait(timeout=2.0)

    executor = router._get_stage_compression_executor(2)
    assert executor.submit(blocker, block=False) is not None
    assert executor.submit(blocker, block=False) is not None
    assert blockers_started.wait(timeout=1.0)

    compression_calls: list[str] = []
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: (
            compression_calls.append(content) or _compression_result(content, "compressed output")
        ),
    )
    try:
        original = _messages(pending=1)
        result = _apply(router, original)
        assert result.messages[1]["content"] == original[1]["content"]
    finally:
        release.set()

    deadline = time.monotonic() + 1.0
    while executor._queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)

    assert compression_calls == []


def test_parallel_exception_cancels_retained_siblings(monkeypatch):
    """A failing compressor must clean up queued siblings before propagating."""
    router = _router()
    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "2")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "2000")
    release = threading.Event()
    executor = router._get_stage_compression_executor(2)
    original_cancel = executor.cancel_pending
    cancelled_batch_sizes: list[int] = []

    def record_cancel(futures):
        cancelled_batch_sizes.append(len(futures))
        return original_cancel(futures)

    monkeypatch.setattr(executor, "cancel_pending", record_cancel)

    def mixed_compress(content, *, context="", bias=1.0):
        if "number 0" in content:
            raise RuntimeError("compressor failed")
        release.wait(timeout=2.0)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", mixed_compress)
    try:
        with pytest.raises(RuntimeError, match="compressor failed"):
            _apply(router, _messages(pending=6))
    finally:
        release.set()

    assert cancelled_batch_sizes
    assert cancelled_batch_sizes[0] >= 1
