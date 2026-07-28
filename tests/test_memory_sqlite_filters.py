"""SQLite memory adapter filter regressions.

- ``valid_at`` point-in-time queries used to AND the default superseded
  filter (``valid_until IS NULL``) with the point-in-time predicate — a
  contradiction for any superseded row, so "what did we know at time t?"
  always returned nothing.
- ``metadata_filters`` with bool/int/float values bound the JSON *text*
  ('true', '3') against json_extract()'s typed SQL value (1, 3), so every
  non-string filter silently matched nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter


@pytest.fixture()
def store(tmp_path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(tmp_path / "memories.db")


async def test_valid_at_returns_superseded_version(store: SQLiteMemoryStore) -> None:
    # Timestamps in this store are naive UTC (see supersede()).
    t0 = datetime.utcnow() - timedelta(hours=2)
    await store.save(Memory(id="a", content="the answer is 41", user_id="u1", valid_from=t0))
    await store.supersede(
        "a",
        Memory(id="b", content="the answer is 42", user_id="u1"),
        supersede_time=datetime.utcnow() - timedelta(hours=1),
    )

    # Present-time query sees only the current version.
    current = await store.query(MemoryFilter(user_id="u1"))
    assert [m.id for m in current] == ["b"]

    # Point-in-time query before the supersession must return the version
    # valid THEN — not an empty set.
    historical = await store.query(
        MemoryFilter(user_id="u1", valid_at=datetime.utcnow() - timedelta(hours=1, minutes=30))
    )
    assert [m.id for m in historical] == ["a"]

    # A point-in-time query after the supersession sees the current version.
    recent = await store.query(MemoryFilter(user_id="u1", valid_at=datetime.utcnow()))
    assert [m.id for m in recent] == ["b"]


async def test_metadata_filters_match_typed_values(store: SQLiteMemoryStore) -> None:
    await store.save(
        Memory(
            id="m1",
            content="prefers dark mode",
            user_id="u1",
            metadata={"active": True, "score": 3, "tag": "x", "weight": 1.5},
        )
    )
    await store.save(
        Memory(
            id="m2",
            content="prefers light mode",
            user_id="u1",
            metadata={"active": False, "score": 7, "tag": "y", "weight": 2.5},
        )
    )

    async def ids(**metadata_filters):
        found = await store.query(MemoryFilter(user_id="u1", metadata_filters=metadata_filters))
        return sorted(m.id for m in found)

    assert await ids(tag="x") == ["m1"]
    assert await ids(active=True) == ["m1"]
    assert await ids(active=False) == ["m2"]
    assert await ids(score=3) == ["m1"]
    assert await ids(score=7) == ["m2"]
    assert await ids(weight=1.5) == ["m1"]
    assert await ids(score=99) == []


async def test_metadata_filters_distinguish_bool_from_number(store: SQLiteMemoryStore) -> None:
    """json_extract collapses JSON true and 1 to the same SQL integer; the
    filter must not — a boolean filter matches only booleans and a numeric
    filter only numbers."""
    await store.save(Memory(id="b1", content="bool flag", user_id="u1", metadata={"flag": True}))
    await store.save(Memory(id="n1", content="numeric flag", user_id="u1", metadata={"flag": 1}))
    await store.save(Memory(id="b0", content="bool off", user_id="u1", metadata={"flag": False}))
    await store.save(Memory(id="n0", content="numeric zero", user_id="u1", metadata={"flag": 0}))

    async def ids(**metadata_filters):
        found = await store.query(MemoryFilter(user_id="u1", metadata_filters=metadata_filters))
        return sorted(m.id for m in found)

    assert await ids(flag=True) == ["b1"]
    assert await ids(flag=1) == ["n1"]
    assert await ids(flag=False) == ["b0"]
    assert await ids(flag=0) == ["n0"]
