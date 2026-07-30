"""Concurrency + cancellation tests for LocalBackend._ensure_initialized.

Per-project backends handed out by ``BackendRouter`` init lazily on first
use. Without a singleflight guard, concurrent first callers each kick off a
parallel init; a slow cold-start (e.g. the ``pytorch_mps`` embedder, >2s)
that is cancelled by an outer ``asyncio.wait_for`` timeout can leave the
backend half-built (``_hierarchical_memory`` still ``None``), tripping the
bare ``assert self._hierarchical_memory is not None`` guards on the retry.

Covers:
- N concurrent first callers trigger exactly one HierarchicalMemory.create.
- A cancelled cold-start resets state so a later call re-inits cleanly.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.memory.adapters.graph import InMemoryGraphStore
from headroom.memory.backends.local import LocalBackend, LocalBackendConfig


def _backend(tmp_path) -> LocalBackend:
    return LocalBackend(
        LocalBackendConfig(
            db_path=str(tmp_path / "memory.db"),
            graph_persist=False,  # InMemoryGraphStore — no SQLite/embedder needed
        )
    )


@pytest.mark.asyncio
async def test_concurrent_ensure_initialized_runs_init_once(tmp_path, monkeypatch):
    hits = {"n": 0}
    release = asyncio.Event()

    async def fake_create(config: Any) -> Any:
        hits["n"] += 1
        # Simulate a slow cold-start so concurrent callers pile up on the
        # lock. Without singleflight, hits would exceed 1.
        await release.wait()
        return MagicMock(name="HierarchicalMemory")

    monkeypatch.setattr("headroom.memory.HierarchicalMemory.create", fake_create)

    backend = _backend(tmp_path)
    tasks = [asyncio.create_task(backend._ensure_initialized()) for _ in range(10)]
    await asyncio.sleep(0)  # let all tasks reach the lock
    release.set()
    await asyncio.gather(*tasks)

    assert hits["n"] == 1
    assert backend._initialized is True
    assert backend._hierarchical_memory is not None


@pytest.mark.asyncio
async def test_cancelled_cold_start_resets_state_and_retries(tmp_path, monkeypatch):
    attempts = {"n": 0}
    first_started = asyncio.Event()
    block_first = asyncio.Event()

    async def fake_create(config: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            first_started.set()
            await block_first.wait()  # never released → this attempt is cancelled
        return MagicMock(name="HierarchicalMemory")

    monkeypatch.setattr("headroom.memory.HierarchicalMemory.create", fake_create)

    backend = _backend(tmp_path)

    # First cold-start gets cancelled by an outer timeout mid-init.
    with pytest.raises((asyncio.TimeoutError, asyncio.CancelledError)):
        await asyncio.wait_for(backend._ensure_initialized(), timeout=0.05)
    await first_started.wait()

    # State must be reset — no half-built backend left behind.
    assert backend._initialized is False
    assert backend._hierarchical_memory is None

    # A subsequent call re-inits cleanly.
    await backend._ensure_initialized()
    assert backend._initialized is True
    assert backend._hierarchical_memory is not None
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_save_memory_accepts_legacy_relation_alias(tmp_path):
    backend = _backend(tmp_path)
    backend._initialized = True
    backend._hierarchical_memory = SimpleNamespace(
        add=AsyncMock(return_value=SimpleNamespace(id="memory-1"))
    )
    backend._graph = InMemoryGraphStore()

    await backend.save_memory(
        content="Alice works at Acme",
        user_id="alice",
        entities=["Alice", "Acme"],
        relationships=[{"source": "Alice", "target": "Acme", "relation": "works_at"}],
    )

    alice = await backend._graph.get_entity_by_name("alice", "Alice")
    assert alice is not None
    relationships = await backend._graph.get_relationships(alice.id)
    assert [relationship.relation_type for relationship in relationships] == ["works_at"]
