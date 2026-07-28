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


async def test_container_filters_do_not_match_stringified_json(store: SQLiteMemoryStore) -> None:
    """json_extract exposes arrays/objects AND strings as SQL text, so an
    array filter must not match a string that holds the same minified JSON
    representation (and vice versa)."""
    await store.save(Memory(id="arr", content="real array", user_id="u1", metadata={"tags": ["a"]}))
    await store.save(
        Memory(id="strfied", content="stringified", user_id="u1", metadata={"tags": '["a"]'})
    )

    async def ids(**metadata_filters):
        found = await store.query(MemoryFilter(user_id="u1", metadata_filters=metadata_filters))
        return sorted(m.id for m in found)

    assert await ids(tags=["a"]) == ["arr"]
    assert await ids(tags='["a"]') == ["strfied"]


async def test_none_filter_requires_explicit_json_null(store: SQLiteMemoryStore) -> None:
    """json_extract maps an explicit JSON null AND a missing key to SQL NULL;
    a None filter must match only memories that explicitly set the key to
    null, not every memory that omits it."""
    await store.save(
        Memory(id="explicit", content="archived: null", user_id="u1", metadata={"archived": None})
    )
    await store.save(Memory(id="missing", content="no archived key", user_id="u1", metadata={}))
    await store.save(
        Memory(id="set", content="archived: x", user_id="u1", metadata={"archived": "x"})
    )

    found = await store.query(MemoryFilter(user_id="u1", metadata_filters={"archived": None}))
    assert sorted(m.id for m in found) == ["explicit"]


async def test_object_filters_match_regardless_of_key_order(store: SQLiteMemoryStore) -> None:
    """JSON objects are unordered: a filter dict built in a different key
    order than the stored metadata must still match (and structural equality
    must stay exact — subset objects or extra keys must not match)."""
    await store.save(
        Memory(id="obj", content="prefs", user_id="u1", metadata={"prefs": {"a": 1, "b": "x"}})
    )
    await store.save(
        Memory(
            id="obj3",
            content="prefs+extra",
            user_id="u1",
            metadata={"prefs": {"a": 1, "b": "x", "c": True}},
        )
    )

    async def ids(**metadata_filters):
        found = await store.query(MemoryFilter(user_id="u1", metadata_filters=metadata_filters))
        return sorted(m.id for m in found)

    # Same pairs, opposite insertion order.
    assert await ids(prefs={"b": "x", "a": 1}) == ["obj"]
    assert await ids(prefs={"a": 1, "b": "x"}) == ["obj"]
    # Subset must not match the 3-key object; the full 3-key filter must.
    assert await ids(prefs={"a": 1, "b": "x", "c": True}) == ["obj3"]
    # Nested type discipline still applies inside objects.
    assert await ids(prefs={"a": True, "b": "x"}) == []


async def test_objects_inside_arrays_match_regardless_of_key_order(
    store: SQLiteMemoryStore,
) -> None:
    """Array order matters, but the key order of objects NESTED in arrays
    must not — elements are compared structurally, per index."""
    await store.save(
        Memory(
            id="arrobj",
            content="steps",
            user_id="u1",
            metadata={"steps": [{"a": 1, "b": 2}, "x"]},
        )
    )

    async def ids(**metadata_filters):
        found = await store.query(MemoryFilter(user_id="u1", metadata_filters=metadata_filters))
        return sorted(m.id for m in found)

    # Nested object in either key order matches.
    assert await ids(steps=[{"b": 2, "a": 1}, "x"]) == ["arrobj"]
    assert await ids(steps=[{"a": 1, "b": 2}, "x"]) == ["arrobj"]
    # Array ORDER still matters; wrong length still fails.
    assert await ids(steps=["x", {"a": 1, "b": 2}]) == []
    assert await ids(steps=[{"a": 1, "b": 2}]) == []


async def test_invalid_nested_key_uses_bounded_comparison(store: SQLiteMemoryStore) -> None:
    """A nested key that can't be expressed as a safe JSON path routes the
    object through the bounded json_each-join comparison (keys are joined as
    DATA, not interpolated into paths): the filter matches exactly the
    intended object — never an arbitrary same-size one."""
    await store.save(
        Memory(id="other", content="unrelated", user_id="u1", metadata={"prefs": {"other": "v"}})
    )
    await store.save(
        Memory(
            id="dotted",
            content="dotted key",
            user_id="u1",
            metadata={"prefs": {"display.name": "alice"}},
        )
    )

    found = await store.query(
        MemoryFilter(user_id="u1", metadata_filters={"prefs": {"display.name": "alice"}})
    )
    assert sorted(m.id for m in found) == ["dotted"]

    found = await store.query(
        MemoryFilter(user_id="u1", metadata_filters={"prefs": {"display.name": "bob"}})
    )
    assert found == []


async def test_large_array_filter_does_not_blow_expression_depth(
    store: SQLiteMemoryStore,
) -> None:
    """A ~500-element array filter must not exceed SQLite's MAX_EXPR_DEPTH;
    past the structural bound it falls back to bounded text equality and
    still matches exactly."""
    big = list(range(500))
    await store.save(Memory(id="big", content="big array", user_id="u1", metadata={"ids": big}))
    await store.save(
        Memory(id="near", content="almost", user_id="u1", metadata={"ids": big[:-1] + [999]})
    )

    async def ids(**metadata_filters):
        found = await store.query(MemoryFilter(user_id="u1", metadata_filters=metadata_filters))
        return sorted(m.id for m in found)

    assert await ids(ids=big) == ["big"]
    assert await ids(ids=big[:-1] + [998]) == []


async def test_deeply_nested_containers_stay_within_expression_depth(
    store: SQLiteMemoryStore,
) -> None:
    """Per-container limits compose multiplicatively: an outer array of eight
    64-element arrays passes every local check but would emit 1000+ ANDs.
    The shared budget must keep the query executable and still exact."""
    nested = [[i * 64 + j for j in range(64)] for i in range(8)]
    await store.save(Memory(id="deep", content="nested", user_id="u1", metadata={"m": nested}))
    almost = [row[:] for row in nested]
    almost[7][63] = -1
    await store.save(Memory(id="near", content="almost", user_id="u1", metadata={"m": almost}))

    async def ids(**metadata_filters):
        found = await store.query(MemoryFilter(user_id="u1", metadata_filters=metadata_filters))
        return sorted(m.id for m in found)

    assert await ids(m=nested) == ["deep"]
    assert await ids(m=almost) == ["near"]


async def test_huge_object_filter_stays_key_order_insensitive(
    store: SQLiteMemoryStore,
) -> None:
    """Above the structural threshold the bounded json_each-join comparison
    keys on entry NAME, so a 65+-key object still matches a filter built in
    reversed insertion order."""
    big = {f"k{i}": i for i in range(70)}
    await store.save(Memory(id="bigobj", content="big", user_id="u1", metadata={"cfg": big}))

    reversed_order = {k: big[k] for k in reversed(list(big))}
    found = await store.query(MemoryFilter(user_id="u1", metadata_filters={"cfg": reversed_order}))
    assert [m.id for m in found] == ["bigobj"]

    wrong = dict(big)
    wrong["k69"] = -1
    found = await store.query(MemoryFilter(user_id="u1", metadata_filters={"cfg": wrong}))
    assert found == []


async def test_bounded_path_matches_nested_objects_and_numerics(
    store: SQLiteMemoryStore,
) -> None:
    """The bounded json_each-join path must stay order-insensitive one level
    below the join and group integer/real as one numeric class — identical
    filters must not flip results with container size."""
    big = {f"k{i}": i for i in range(66)}
    big["nested"] = {"x": 1, "y": 2}
    big["ratio"] = 1.0  # stored as JSON real
    await store.save(Memory(id="bigobj", content="big", user_id="u1", metadata={"cfg": big}))

    flt = dict(big)
    flt["nested"] = {"y": 2, "x": 1}  # reversed key order below the join
    flt["ratio"] = 1  # int filter vs stored real
    found = await store.query(MemoryFilter(user_id="u1", metadata_filters={"cfg": flt}))
    assert [m.id for m in found] == ["bigobj"]

    wrong = dict(flt)
    wrong["nested"] = {"y": 3, "x": 1}
    found = await store.query(MemoryFilter(user_id="u1", metadata_filters={"cfg": wrong}))
    assert found == []
