"""Durable memory aliases (PR #15 Codex review, P1).

Passive recall renders a short handle instead of the full backend UUID so
``memory_update`` / ``memory_delete`` can address a row without spending
~16 prompt tokens per ID (docs/token-efficiency-review.md F4).

The first implementation minted *session-local* counters (``m1``, ``m2``, …)
held in per-handler dicts. That is a data-integrity bug:

* After a proxy restart — or when consecutive requests land on different
  workers — the map is empty while ``[m1]`` references are still live in the
  client's transcript, so the operation fails.
* Worse, first-seen ordering could assign ``m1`` to a *different* memory in
  the new process, so ``memory_update`` / ``memory_delete`` would silently
  mutate the wrong persistent record.

The fix makes the alias a pure function of the memory's own ID (``m:`` plus
its first 8 characters) and resolves it server-side by prefix lookup against
the backend. These tests pin the properties that make that safe:

1. A *fresh* handler (empty process state) resolves an alias minted by
   another handler instance.
2. Ambiguity fails loudly instead of mutating a guessed record.
3. A stale alias (memory deleted) fails loudly.
4. Full IDs pass through untouched.
5. Two rows in the same block that would share an alias both fall back to
   their full IDs, so what we render is never ambiguous.

The PR #16 review then found that resolution enumerated only a bounded
10,000-row slice, so a recalled memory outside that slice could be aliased but
never addressed — and, worse, a lone match *inside* a truncated slice was
treated as unique when a second match could be sitting past the cut. The last
two sections pin the repaired behaviour:

6. A memory past the first page still resolves (the scan escalates until the
   listing is provably exhausted).
7. Anything short of proof — a listing that never exhausts, a backend that
   cannot list at all — raises instead of guessing.
8. A backend that can answer the prefix question itself short-circuits the
   enumeration entirely.

The third finding in the same scheme: backend IDs are opaque and may begin with
``m:`` themselves, and ``_alias_for_memory`` emits short IDs verbatim, so such
an ID reaches the model looking exactly like an alias. Reading it as one — strip
the sigil, prefix-match the remainder — rejected the real record, or resolved to
a *different* record whose ID began with the remainder and mutated it. So
resolution is no longer prefix matching: it collects every stored memory that
would have been *rendered* as the token (its native id, or its alias) and still
insists on exactly one. The last section pins that:

9. A native id beginning with ``m:`` round-trips through update and delete to
   itself; a short one that is byte-identical to another memory's alias is
   reported as ambiguous instead of picking one; and neither reading is
   prefix-matched.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from headroom.proxy import memory_handler
from headroom.proxy.memory_handler import (
    MEMORY_ALIAS_PREFIX,
    MemoryAliasError,
    MemoryConfig,
    MemoryHandler,
    MemoryMode,
)

# ---------------------------------------------------------------------------
# Stub backend
# ---------------------------------------------------------------------------


@dataclass
class _Memory:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Result:
    memory: _Memory
    score: float = 0.9
    related_entities: list[str] = field(default_factory=list)


class _Backend:
    """Minimal backend exposing the listing API alias resolution uses."""

    def __init__(self, memories: list[_Memory]) -> None:
        self.memories = list(memories)
        self.updated: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    async def get_user_memories(self, user_id: str, limit: int = 100) -> list[_Memory]:
        return list(self.memories[:limit])

    async def search_memories(self, query: str = "", **kwargs: Any) -> list[_Result]:
        top_k = kwargs.get("top_k", 10)
        return [_Result(memory=m) for m in self.memories[:top_k]]

    async def update_memory(self, memory_id: str, new_content: str, **kwargs: Any) -> _Memory:
        for m in self.memories:
            if m.id == memory_id:
                m.content = new_content
                self.updated.append((memory_id, new_content))
                return m
        raise KeyError(memory_id)

    async def delete_memory(self, memory_id: str) -> bool:
        for i, m in enumerate(self.memories):
            if m.id == memory_id:
                del self.memories[i]
                self.deleted.append(memory_id)
                return True
        return False


def _handler(backend: _Backend) -> MemoryHandler:
    handler = MemoryHandler(
        MemoryConfig(
            enabled=True,
            backend="local",
            inject_context=True,
            inject_tools=True,
            top_k=5,
            min_similarity=0.3,
            mode=MemoryMode.AUTO_TAIL,
        )
    )
    handler._backend = backend  # type: ignore[assignment]
    handler._initialized = True
    return handler


UUID_A = "a1b2c3d4-1111-4aaa-8bbb-000000000001"
UUID_B = "f9e8d7c6-2222-4ccc-8ddd-000000000002"


def _fresh_backend() -> _Backend:
    return _Backend(
        [
            _Memory(id=UUID_A, content="User prefers Python."),
            _Memory(id=UUID_B, content="User is in America/Los_Angeles."),
        ]
    )


# ---------------------------------------------------------------------------
# The alias itself
# ---------------------------------------------------------------------------


def test_alias_is_a_pure_function_of_the_memory_id() -> None:
    """No handler state is involved — two unrelated instances agree.

    This is what makes the alias survive a restart: it is derived, not
    assigned, so there is no map to lose.
    """
    a = _handler(_fresh_backend())
    b = _handler(_fresh_backend())
    assert a._alias_for_memory(UUID_A) == f"{MEMORY_ALIAS_PREFIX}a1b2c3d4"
    assert a._alias_for_memory(UUID_A) == b._alias_for_memory(UUID_A)
    assert a._alias_for_memory(UUID_A) != a._alias_for_memory(UUID_B)


def test_short_ids_are_not_aliased() -> None:
    """An alias longer than the ID saves nothing, so render the ID."""
    handler = _handler(_fresh_backend())
    assert handler._alias_for_memory("abc123") == "abc123"


def test_handler_keeps_no_alias_map() -> None:
    """Regression guard: any alias→id map on the handler would be an
    unshared source of truth across workers and restarts."""
    handler = _handler(_fresh_backend())
    assert not hasattr(handler, "_memory_id_by_alias")
    assert not hasattr(handler, "_alias_by_memory_id")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_fresh_process_resolves_alias_minted_elsewhere() -> None:
    """THE regression test for the review finding.

    A handler that never rendered the recall block (a restarted proxy, or a
    different worker) must still resolve ``m:a1b2c3d4`` to the memory it was
    minted for.
    """
    minting = _handler(_fresh_backend())
    alias = minting._alias_for_memory(UUID_A)

    # Brand new handler, brand new backend object: no shared state at all.
    fresh_backend = _fresh_backend()
    fresh = _handler(fresh_backend)
    resolved = asyncio.run(fresh._resolve_memory_alias(fresh_backend, "u", alias))
    assert resolved == UUID_A


def test_alias_resolution_survives_reordered_results() -> None:
    """First-seen ordering must not matter: the old counter scheme assigned
    ``m1`` to whichever memory came back first, so a re-ordered result set
    silently re-pointed the alias."""
    backend = _Backend(
        [
            _Memory(id=UUID_B, content="second first now"),
            _Memory(id=UUID_A, content="first second now"),
        ]
    )
    handler = _handler(backend)
    alias = handler._alias_for_memory(UUID_A)
    assert asyncio.run(handler._resolve_memory_alias(backend, "u", alias)) == UUID_A


def test_full_ids_pass_through_unchanged() -> None:
    """IDs from memory_search / memory_list keep working verbatim."""
    backend = _fresh_backend()
    handler = _handler(backend)
    assert asyncio.run(handler._resolve_memory_alias(backend, "u", UUID_A)) == UUID_A
    # A value that is neither an alias nor a stored ID is still passed
    # through — the backend decides whether it exists.
    assert asyncio.run(handler._resolve_memory_alias(backend, "u", "nope")) == "nope"


def test_ambiguous_alias_raises_instead_of_guessing() -> None:
    """Two memories sharing the alias prefix must abort the operation."""
    backend = _Backend(
        [
            _Memory(id="dupprefix-1111-4aaa-8bbb-000000000001", content="one"),
            _Memory(id="dupprefix-2222-4ccc-8ddd-000000000002", content="two"),
        ]
    )
    handler = _handler(backend)
    alias = handler._alias_for_memory("dupprefix-1111-4aaa-8bbb-000000000001")
    with pytest.raises(MemoryAliasError, match="ambiguous"):
        asyncio.run(handler._resolve_memory_alias(backend, "u", alias))


def test_unknown_alias_raises() -> None:
    """A stale alias (memory since deleted) fails loudly."""
    backend = _fresh_backend()
    handler = _handler(backend)
    with pytest.raises(MemoryAliasError, match="No memory matches"):
        asyncio.run(handler._resolve_memory_alias(backend, "u", f"{MEMORY_ALIAS_PREFIX}deadbeef"))


def test_empty_alias_raises() -> None:
    backend = _fresh_backend()
    handler = _handler(backend)
    with pytest.raises(MemoryAliasError, match="Malformed"):
        asyncio.run(handler._resolve_memory_alias(backend, "u", MEMORY_ALIAS_PREFIX))


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def test_update_via_alias_hits_the_right_record() -> None:
    backend = _fresh_backend()
    handler = _handler(backend)
    alias = handler._alias_for_memory(UUID_B)

    raw = asyncio.run(
        handler._execute_update({"memory_id": alias, "new_content": "moved to UTC"}, "u")
    )
    assert json.loads(raw)["status"] == "updated"
    assert backend.updated == [(UUID_B, "moved to UTC")]


def test_delete_via_alias_hits_the_right_record() -> None:
    backend = _fresh_backend()
    handler = _handler(backend)
    alias = handler._alias_for_memory(UUID_A)

    raw = asyncio.run(handler._execute_delete({"memory_id": alias}, "u"))
    payload = json.loads(raw)
    assert payload["status"] == "deleted"
    assert payload["memory_id"] == UUID_A
    assert backend.deleted == [UUID_A]


def test_update_with_ambiguous_alias_errors_and_mutates_nothing() -> None:
    """The load-bearing safety property: on ambiguity the record is left
    alone and the model gets an actionable error."""
    backend = _Backend(
        [
            _Memory(id="dupprefix-1111-4aaa-8bbb-000000000001", content="one"),
            _Memory(id="dupprefix-2222-4ccc-8ddd-000000000002", content="two"),
        ]
    )
    handler = _handler(backend)
    alias = handler._alias_for_memory("dupprefix-1111-4aaa-8bbb-000000000001")

    raw = asyncio.run(
        handler._execute_update({"memory_id": alias, "new_content": "clobbered"}, "u")
    )
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert "ambiguous" in payload["error"]
    assert backend.updated == []
    assert [m.content for m in backend.memories] == ["one", "two"]


def test_delete_with_unknown_alias_errors_and_deletes_nothing() -> None:
    backend = _fresh_backend()
    handler = _handler(backend)

    raw = asyncio.run(handler._execute_delete({"memory_id": f"{MEMORY_ALIAS_PREFIX}deadbeef"}, "u"))
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert backend.deleted == []
    assert len(backend.memories) == 2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_recall_block_renders_durable_aliases() -> None:
    backend = _fresh_backend()
    handler = _handler(backend)
    context = asyncio.run(
        handler.search_and_format_context("u", [{"role": "user", "content": "hi"}])
    )
    assert context is not None
    assert f"[{MEMORY_ALIAS_PREFIX}a1b2c3d4]" in context
    assert f"[{MEMORY_ALIAS_PREFIX}f9e8d7c6]" in context
    assert UUID_A not in context


def test_colliding_rows_render_full_ids() -> None:
    """If two rendered rows would share an alias, both fall back to the full
    ID — what we render is never ambiguous by construction."""
    ids = [
        "dupprefix-1111-4aaa-8bbb-000000000001",
        "dupprefix-2222-4ccc-8ddd-000000000002",
        UUID_A,
    ]
    backend = _Backend([_Memory(id=i, content=f"row {i}") for i in ids])
    handler = _handler(backend)
    context = asyncio.run(
        handler.search_and_format_context("u", [{"role": "user", "content": "hi"}])
    )
    assert context is not None
    assert f"[{ids[0]}]" in context
    assert f"[{ids[1]}]" in context
    # The non-colliding row still gets the cheap alias.
    assert f"[{MEMORY_ALIAS_PREFIX}a1b2c3d4]" in context


def test_alias_row_ids_handles_missing_ids() -> None:
    assert MemoryHandler._alias_row_ids([UUID_A, "", UUID_B]) == [
        f"{MEMORY_ALIAS_PREFIX}a1b2c3d4",
        "?",
        f"{MEMORY_ALIAS_PREFIX}f9e8d7c6",
    ]


# ---------------------------------------------------------------------------
# Reach: resolution must not be capped at a bounded slice (PR #16 review, P2)
# ---------------------------------------------------------------------------


class _PagedBackend(_Backend):
    """Backend whose listing honours ``limit`` and records what was asked."""

    def __init__(self, memories: list[_Memory]) -> None:
        super().__init__(memories)
        self.limits: list[int] = []

    async def get_user_memories(self, user_id: str, limit: int = 100) -> list[_Memory]:
        self.limits.append(limit)
        return list(self.memories[:limit])


def _many(count: int, target_index: int, target_id: str) -> list[_Memory]:
    memories = [
        _Memory(id=f"{i:08x}-0000-4000-8000-{i:012d}", content=f"m{i}") for i in range(count)
    ]
    memories[target_index] = _Memory(id=target_id, content="the recalled one")
    return memories


def test_alias_resolves_for_a_memory_beyond_the_first_page() -> None:
    """THE regression test for the review finding.

    Passive recall can surface (and therefore alias) any stored memory. When
    the record sits past the old hard-capped 10,000-row lookup slice, update /
    delete used to report "no memory matches" for an alias the model had just
    been shown. The scan now escalates until the listing is exhausted.
    """
    target = "beef1234-9999-4fff-8fff-999999999999"
    backend = _PagedBackend(_many(12_001, 12_000, target))
    handler = _handler(backend)
    alias = handler._alias_for_memory(target)

    assert asyncio.run(handler._resolve_memory_alias(backend, "u", alias)) == target
    # It had to look past the first page to get there.
    assert backend.limits[0] == 10_000
    assert max(backend.limits) > 10_000


def test_ordinary_store_is_listed_exactly_once() -> None:
    """Escalation must not cost a second round-trip in the normal case: a
    listing shorter than the page size has already proven itself complete."""
    backend = _PagedBackend(_fresh_backend().memories)
    handler = _handler(backend)
    asyncio.run(handler._resolve_memory_alias(backend, "u", handler._alias_for_memory(UUID_A)))
    assert backend.limits == [10_000]


def test_delete_reaches_a_memory_beyond_the_first_page() -> None:
    """End to end: the advertised ``[id]`` is actually usable."""
    target = "beef1234-9999-4fff-8fff-999999999999"
    backend = _PagedBackend(_many(12_001, 12_000, target))
    handler = _handler(backend)

    raw = asyncio.run(
        handler._execute_delete({"memory_id": handler._alias_for_memory(target)}, "u")
    )
    payload = json.loads(raw)
    assert payload["status"] == "deleted"
    assert payload["memory_id"] == target
    assert backend.deleted == [target]


def test_single_match_in_a_truncated_listing_is_never_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safety property, restated for the capped case.

    A listing that keeps returning exactly as many rows as were asked for is
    not proof of anything: a second memory sharing the prefix could be sitting
    just past the cut. Resolving the single visible match would mutate a
    guessed record, so resolution must fail instead.
    """
    monkeypatch.setattr(memory_handler, "MEMORY_ALIAS_PAGE_LIMIT", 4)
    monkeypatch.setattr(memory_handler, "MEMORY_ALIAS_MAX_LOOKUP", 16)

    class _AlwaysTruncating(_Backend):
        async def get_user_memories(self, user_id: str, limit: int = 100) -> list[_Memory]:
            # Always "full": never proves it returned everything.
            return [_Memory(id=UUID_A, content="visible")] + [
                _Memory(id=f"{i:08x}-1111-4111-8111-{i:012d}", content="filler")
                for i in range(limit - 1)
            ]

    backend = _AlwaysTruncating([])
    handler = _handler(backend)
    with pytest.raises(MemoryAliasError, match="cannot be proven"):
        asyncio.run(handler._resolve_memory_alias(backend, "u", handler._alias_for_memory(UUID_A)))


def test_backend_without_a_listing_api_reports_that_it_cannot_resolve() -> None:
    """A backend that cannot enumerate must not be reported as "deleted"."""

    class _Opaque:
        async def delete_memory(self, memory_id: str) -> bool:
            raise AssertionError("must not be reached")

    handler = _handler(_fresh_backend())
    backend = _Opaque()
    with pytest.raises(MemoryAliasError, match="cannot"):
        asyncio.run(handler._resolve_memory_alias(backend, "u", f"{MEMORY_ALIAS_PREFIX}a1b2c3d4"))


def test_semantic_search_is_not_used_to_prove_alias_uniqueness() -> None:
    """A short semantic result is not proof that every colliding ID was seen."""

    class _SearchOnly:
        async def search_memories(self, **kwargs: Any) -> list[_Result]:
            return [_Result(memory=_Memory(id=UUID_A, content="visible match"))]

    handler = _handler(_fresh_backend())
    with pytest.raises(MemoryAliasError, match="cannot list"):
        asyncio.run(
            handler._resolve_memory_alias(
                _SearchOnly(),
                "u",
                handler._alias_for_memory(UUID_A),
            )
        )


# ---------------------------------------------------------------------------
# Exact backend-side prefix lookup
# ---------------------------------------------------------------------------


class _PrefixCapableBackend(_Backend):
    """Backend that answers the prefix question itself — no enumeration."""

    def __init__(self, memories: list[_Memory]) -> None:
        super().__init__(memories)
        self.enumerated = False

    async def get_memory_ids_by_prefix(self, prefix: str, user_id: str) -> list[_Memory]:
        return [m for m in self.memories if m.id.startswith(prefix)]

    async def get_user_memories(self, user_id: str, limit: int = 100) -> list[_Memory]:
        self.enumerated = True
        return []


def test_backend_prefix_hook_short_circuits_enumeration() -> None:
    backend = _PrefixCapableBackend(_fresh_backend().memories)
    handler = _handler(backend)
    alias = handler._alias_for_memory(UUID_B)

    assert asyncio.run(handler._resolve_memory_alias(backend, "u", alias)) == UUID_B
    assert backend.enumerated is False


def test_backend_prefix_hook_still_fails_loudly_on_ambiguity() -> None:
    backend = _PrefixCapableBackend(
        [
            _Memory(id="dupprefix-1111-4aaa-8bbb-000000000001", content="one"),
            _Memory(id="dupprefix-2222-4ccc-8ddd-000000000002", content="two"),
        ]
    )
    handler = _handler(backend)
    alias = handler._alias_for_memory("dupprefix-1111-4aaa-8bbb-000000000001")
    with pytest.raises(MemoryAliasError, match="ambiguous"):
        asyncio.run(handler._resolve_memory_alias(backend, "u", alias))


# ---------------------------------------------------------------------------
# Native IDs that themselves begin with the alias sigil (PR #16 review, P2)
#
# Backend IDs are opaque strings; nothing stops one from starting with ``m:``.
# ``memory_search`` / ``memory_list`` report native IDs verbatim, and
# ``_alias_for_memory`` hands short ones straight back, so such an ID reaches
# the model looking exactly like an alias. Reading it as one unconditionally —
# strip the sigil, prefix-match the rest — either rejected the real record or,
# when a *different* ID happened to start with the remainder, resolved to that
# other record and mutated it. That is the same failure mode as the two earlier
# defects in this scheme, so resolution now asks which memories would have been
# *rendered* as the token (native id or alias) and insists the answer is unique.
# ---------------------------------------------------------------------------

# A short native ID that is indistinguishable from an alias: exactly the shape
# ``_alias_for_memory`` emits, and short enough that it is emitted verbatim.
SIGIL_SHORT = f"{MEMORY_ALIAS_PREFIX}beef1234"
# A long native ID that also happens to start with the sigil.
SIGIL_LONG = f"{MEMORY_ALIAS_PREFIX}abc12345-1111-4aaa-8bbb-000000000003"
# A decoy whose ID begins with SIGIL_SHORT's *body*: the record the old
# strip-and-prefix-match resolver would have hit instead.
DECOY = "beef1234-0000-4000-8000-000000000002"
# The same trap for SIGIL_LONG: a *different* memory whose ID is exactly the
# sigil-stripped remainder, so the old resolver found it as the unique prefix
# match and edited it in place of the record actually addressed.
LONG_DECOY = SIGIL_LONG[len(MEMORY_ALIAS_PREFIX) :]


def test_sigil_leading_short_id_is_rendered_verbatim() -> None:
    """The self-collision the review names: the alias function returns a short
    ``m:``-leading ID as itself, so it comes back looking like an alias."""
    assert MemoryHandler._alias_for_memory(SIGIL_SHORT) == SIGIL_SHORT


def test_native_id_beginning_with_sigil_resolves_to_itself() -> None:
    backend = _Backend(
        [_Memory(id=SIGIL_SHORT, content="native short"), _Memory(id=UUID_A, content="unrelated")]
    )
    handler = _handler(backend)
    assert asyncio.run(handler._resolve_memory_alias(backend, "u", SIGIL_SHORT)) == SIGIL_SHORT


def test_long_native_id_beginning_with_sigil_resolves_to_itself() -> None:
    backend = _Backend(
        [_Memory(id=SIGIL_LONG, content="native long"), _Memory(id=UUID_A, content="unrelated")]
    )
    handler = _handler(backend)
    assert asyncio.run(handler._resolve_memory_alias(backend, "u", SIGIL_LONG)) == SIGIL_LONG


def test_long_sigil_id_resolves_to_itself_not_its_stripped_twin() -> None:
    """The sharpest form of the finding, and it is not merely a failed lookup.

    ``m:abc12345-…`` and ``abc12345-…`` are two different memories. Stripping
    the sigil turns the first reference into the second, and the second was the
    unique prefix match, so ``memory_update`` silently rewrote the wrong record.
    Render-equality has only one candidate here, so this resolves *correctly*
    rather than merely refusing.
    """
    backend = _Backend(
        [
            _Memory(id=SIGIL_LONG, content="the addressed one"),
            _Memory(id=LONG_DECOY, content="the stripped twin"),
        ]
    )
    handler = _handler(backend)

    raw = asyncio.run(
        handler._execute_update({"memory_id": SIGIL_LONG, "new_content": "edited"}, "u")
    )
    assert json.loads(raw)["status"] == "updated"
    assert backend.updated == [(SIGIL_LONG, "edited")]
    assert [m.content for m in backend.memories] == ["edited", "the stripped twin"]


def test_update_with_sigil_leading_native_id_hits_that_record() -> None:
    """Round-trip: memory_list hands back ``m:beef1234``, memory_update must
    edit *that* memory."""
    backend = _Backend(
        [
            _Memory(id=SIGIL_SHORT, content="native short"),
            _Memory(id=UUID_A, content="unrelated"),
        ]
    )
    handler = _handler(backend)

    raw = asyncio.run(
        handler._execute_update({"memory_id": SIGIL_SHORT, "new_content": "edited"}, "u")
    )
    assert json.loads(raw)["status"] == "updated"
    assert backend.updated == [(SIGIL_SHORT, "edited")]
    assert [m.content for m in backend.memories] == ["edited", "unrelated"]


def test_delete_with_sigil_leading_native_id_hits_that_record() -> None:
    backend = _Backend(
        [
            _Memory(id=SIGIL_LONG, content="native long"),
            _Memory(id=UUID_A, content="unrelated"),
        ]
    )
    handler = _handler(backend)

    raw = asyncio.run(handler._execute_delete({"memory_id": SIGIL_LONG}, "u"))
    payload = json.loads(raw)
    assert payload["status"] == "deleted"
    assert payload["memory_id"] == SIGIL_LONG
    assert backend.deleted == [SIGIL_LONG]
    assert [m.id for m in backend.memories] == [UUID_A]


def test_sigil_leading_native_id_never_mutates_the_prefix_neighbour() -> None:
    """THE regression test for the review finding.

    ``m:beef1234`` is a real memory, and ``beef1234-…`` is a *different* real
    memory whose ID starts with the sigil-stripped remainder. The old resolver
    found exactly one prefix match — the neighbour — and silently updated it.
    Both readings are legitimate here, so the only correct answer is to refuse.
    """
    backend = _Backend(
        [
            _Memory(id=SIGIL_SHORT, content="native short"),
            _Memory(id=DECOY, content="innocent bystander"),
        ]
    )
    handler = _handler(backend)

    raw = asyncio.run(
        handler._execute_update({"memory_id": SIGIL_SHORT, "new_content": "clobbered"}, "u")
    )
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert "ambiguous" in payload["error"]
    assert backend.updated == []
    assert [m.content for m in backend.memories] == ["native short", "innocent bystander"]


def test_sigil_leading_native_id_never_deletes_the_prefix_neighbour() -> None:
    backend = _Backend(
        [
            _Memory(id=SIGIL_SHORT, content="native short"),
            _Memory(id=DECOY, content="innocent bystander"),
        ]
    )
    handler = _handler(backend)

    raw = asyncio.run(handler._execute_delete({"memory_id": SIGIL_SHORT}, "u"))
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert "ambiguous" in payload["error"]
    assert backend.deleted == []
    assert len(backend.memories) == 2


def test_alias_still_wins_when_no_native_id_claims_the_token() -> None:
    """The neighbour alone is unambiguous: the token is only its alias."""
    backend = _Backend([_Memory(id=DECOY, content="the only claimant")])
    handler = _handler(backend)
    assert asyncio.run(handler._resolve_memory_alias(backend, "u", SIGIL_SHORT)) == DECOY


def test_prefix_hook_finds_a_sigil_leading_native_id() -> None:
    """The backend-side short-circuit must probe both readings too, or it
    reintroduces the bug for backends that implement the hook."""
    backend = _PrefixCapableBackend(
        [_Memory(id=SIGIL_SHORT, content="native"), _Memory(id=UUID_A, content="unrelated")]
    )
    handler = _handler(backend)
    assert asyncio.run(handler._resolve_memory_alias(backend, "u", SIGIL_SHORT)) == SIGIL_SHORT
    assert backend.enumerated is False


def test_prefix_hook_refuses_when_both_readings_claim_the_token() -> None:
    backend = _PrefixCapableBackend(
        [_Memory(id=SIGIL_SHORT, content="native"), _Memory(id=DECOY, content="neighbour")]
    )
    handler = _handler(backend)
    with pytest.raises(MemoryAliasError, match="ambiguous"):
        asyncio.run(handler._resolve_memory_alias(backend, "u", SIGIL_SHORT))


def test_partial_alias_is_not_prefix_matched() -> None:
    """Resolution is render-equality, not prefix matching.

    Every minted alias is exactly ``m:`` + 8 characters, so a shorter token was
    never rendered by us; prefix-matching it would be a guess, and guessing is
    how the previous two defects mutated the wrong record.
    """
    backend = _fresh_backend()
    handler = _handler(backend)
    with pytest.raises(MemoryAliasError, match="No memory matches"):
        asyncio.run(handler._resolve_memory_alias(backend, "u", f"{MEMORY_ALIAS_PREFIX}a1b2"))


def test_sigil_leading_native_id_is_unaddressable_but_never_wrong_on_opaque_backends() -> None:
    """A backend that cannot enumerate cannot prove which reading is meant.

    Passing the token through verbatim would "work" most of the time and hit
    the wrong record the rest — the exact trade the alias scheme refuses.
    """

    class _Opaque:
        async def delete_memory(self, memory_id: str) -> bool:
            raise AssertionError("must not be reached")

    handler = _handler(_fresh_backend())
    with pytest.raises(MemoryAliasError, match="cannot"):
        asyncio.run(handler._resolve_memory_alias(_Opaque(), "u", SIGIL_SHORT))
