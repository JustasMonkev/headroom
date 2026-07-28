"""SQLite memory store for Headroom's hierarchical memory system.

Provides persistent storage for Memory objects with full support for:
- Hierarchical scope filtering (user/session/agent/turn)
- Temporal versioning with supersession chains
- Point-in-time queries
- Efficient batch operations
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import Memory, ScopeLevel
from ..ports import MemoryFilter

if TYPE_CHECKING:
    import numpy as np

# Regex pattern for safe metadata keys: alphanumeric, underscores, hyphens only
# This prevents JSON path injection attacks via malicious key names
_SAFE_METADATA_KEY_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\-]*$")

# Container filters above this many entries fall back from per-entry SQL
# predicates to the bounded json_each-join comparison: each entry emits 1-2
# AND predicates, and a ~500-element array would exceed SQLite's
# MAX_EXPR_DEPTH (default 1000), failing the whole query with "Expression
# tree is too large".
_MAX_STRUCTURAL_ITEMS = 64

# Shared budget for the TOTAL number of predicates one metadata_filters build
# may emit: per-container limits alone compose multiplicatively (an array of
# eight 64-element arrays passes every local check yet emits 1000+ ANDs).
# Once exhausted, remaining containers use the bounded comparison.
_MAX_METADATA_PREDICATES = 256


def _validate_metadata_key(key: str) -> bool:
    """Validate that a metadata key is safe for use in JSON path expressions.

    Prevents JSON path injection by ensuring keys contain only safe characters.
    Valid keys: start with letter or underscore, contain only alphanumeric, underscore, hyphen.

    Args:
        key: The metadata key to validate.

    Returns:
        True if the key is safe, False otherwise.
    """
    if not key or len(key) > 255:
        return False
    return _SAFE_METADATA_KEY_PATTERN.match(key) is not None


class SQLiteMemoryStore:
    """SQLite-based memory store implementing the MemoryStore protocol.

    Features:
    - Full CRUD operations with batch support
    - Hierarchical scope filtering (user -> session -> agent -> turn)
    - Temporal versioning with supersession chains
    - Point-in-time queries via valid_at filter
    - Thread-safe: connection-per-request pattern

    Usage:
        store = SQLiteMemoryStore("./memories.db")
        await store.save(memory)
        memories = await store.query(MemoryFilter(user_id="alice"))

    Schema:
        The memories table stores all Memory fields with appropriate indexes
        for efficient querying by scope, category, importance, and time.
    """

    def __init__(self, db_path: str | Path = "headroom_memory.db") -> None:
        """Initialize the SQLite memory store.

        Args:
            db_path: Path to SQLite database file. Created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection (thread-safe pattern).

        Returns:
            A new SQLite connection with row factory configured.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the database schema with indexes."""
        with self._get_conn() as conn:
            # Create memories table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,

                    -- Hierarchical scoping
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    agent_id TEXT,
                    turn_id TEXT,

                    -- Temporal
                    created_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,

                    -- Classification
                    category TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,

                    -- Lineage
                    supersedes TEXT,
                    superseded_by TEXT,
                    promoted_from TEXT,
                    promotion_chain TEXT NOT NULL DEFAULT '[]',

                    -- Access tracking
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT,

                    -- Entity references (JSON array)
                    entity_refs TEXT NOT NULL DEFAULT '[]',

                    -- Embedding (BLOB for numpy array)
                    embedding BLOB,

                    -- Metadata (JSON object)
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)

            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_turn_id ON memories(turn_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_valid_until ON memories(valid_until)"
            )

            # Composite index for common scope queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(user_id, session_id, agent_id, turn_id)
            """)

            # Index for supersession chain traversal
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories(supersedes)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_superseded_by ON memories(superseded_by)"
            )

            conn.commit()

    def _serialize_embedding(self, embedding: np.ndarray | None) -> bytes | None:
        """Serialize numpy array to bytes for BLOB storage."""
        if embedding is None:
            return None
        import numpy as np

        return bytes(embedding.astype(np.float32).tobytes())

    def _deserialize_embedding(
        self, data: bytes | None, dim: int | None = None
    ) -> np.ndarray | None:
        """Deserialize bytes back to numpy array."""
        if data is None:
            return None
        import numpy as np

        arr = np.frombuffer(data, dtype=np.float32)
        return arr

    def _memory_to_row(self, memory: Memory) -> dict[str, Any]:
        """Convert Memory object to row dict for insertion."""
        return {
            "id": memory.id,
            "content": memory.content,
            "user_id": memory.user_id,
            "session_id": memory.session_id,
            "agent_id": memory.agent_id,
            "turn_id": memory.turn_id,
            "created_at": memory.created_at.isoformat(),
            "valid_from": memory.valid_from.isoformat(),
            "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
            "category": "",  # Deprecated - kept for backwards compatibility
            "importance": memory.importance,
            "supersedes": memory.supersedes,
            "superseded_by": memory.superseded_by,
            "promoted_from": memory.promoted_from,
            "promotion_chain": json.dumps(memory.promotion_chain),
            "access_count": memory.access_count,
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "entity_refs": json.dumps(memory.entity_refs),
            "embedding": self._serialize_embedding(memory.embedding),
            "metadata": json.dumps(memory.metadata),
        }

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            turn_id=row["turn_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            importance=row["importance"],
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            promoted_from=row["promoted_from"],
            promotion_chain=json.loads(row["promotion_chain"]) if row["promotion_chain"] else [],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"])
            if row["last_accessed"]
            else None,
            entity_refs=json.loads(row["entity_refs"]) if row["entity_refs"] else [],
            embedding=self._deserialize_embedding(row["embedding"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def save(self, memory: Memory) -> None:
        """Save or update a memory.

        If a memory with the same ID exists, it will be updated.

        Args:
            memory: The memory to save.
        """
        row = self._memory_to_row(memory)

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, content, user_id, session_id, agent_id, turn_id,
                    created_at, valid_from, valid_until,
                    category, importance,
                    supersedes, superseded_by, promoted_from, promotion_chain,
                    access_count, last_accessed,
                    entity_refs, embedding, metadata
                ) VALUES (
                    :id, :content, :user_id, :session_id, :agent_id, :turn_id,
                    :created_at, :valid_from, :valid_until,
                    :category, :importance,
                    :supersedes, :superseded_by, :promoted_from, :promotion_chain,
                    :access_count, :last_accessed,
                    :entity_refs, :embedding, :metadata
                )
                """,
                row,
            )
            conn.commit()

    async def save_batch(self, memories: list[Memory]) -> None:
        """Save multiple memories in a single transaction.

        Args:
            memories: List of memories to save.
        """
        if not memories:
            return

        rows = [self._memory_to_row(m) for m in memories]

        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO memories (
                    id, content, user_id, session_id, agent_id, turn_id,
                    created_at, valid_from, valid_until,
                    category, importance,
                    supersedes, superseded_by, promoted_from, promotion_chain,
                    access_count, last_accessed,
                    entity_refs, embedding, metadata
                ) VALUES (
                    :id, :content, :user_id, :session_id, :agent_id, :turn_id,
                    :created_at, :valid_from, :valid_until,
                    :category, :importance,
                    :supersedes, :superseded_by, :promoted_from, :promotion_chain,
                    :access_count, :last_accessed,
                    :entity_refs, :embedding, :metadata
                )
                """,
                rows,
            )
            conn.commit()

    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            The memory if found, None otherwise.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_memory(row)

    async def get_batch(self, memory_ids: list[str]) -> list[Memory]:
        """Retrieve multiple memories by ID.

        Args:
            memory_ids: List of memory IDs to retrieve.

        Returns:
            List of found memories (may be shorter than input if some not found).
        """
        if not memory_ids:
            return []

        placeholders = ", ".join("?" * len(memory_ids))

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",  # nosec B608
                memory_ids,
            )

            return [self._row_to_memory(row) for row in cursor]

    async def record_access(
        self,
        memory_ids: list[str],
        accessed_at: datetime | None = None,
    ) -> int:
        """Atomically record one retrieval for each distinct memory ID."""
        unique_ids = list(dict.fromkeys(memory_ids))
        if not unique_ids:
            return 0

        timestamp = accessed_at or datetime.utcnow()
        placeholders = ", ".join("?" for _ in unique_ids)
        with self._get_conn() as conn:
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id IN ({placeholders})
                """,  # nosec B608
                [timestamp.isoformat(), *unique_ids],
            )
            conn.commit()
            return cursor.rowcount

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if the memory was deleted, False if not found.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def delete_batch(self, memory_ids: list[str]) -> int:
        """Delete multiple memories by ID.

        Args:
            memory_ids: List of memory IDs to delete.

        Returns:
            Number of memories actually deleted.
        """
        if not memory_ids:
            return 0

        placeholders = ", ".join("?" * len(memory_ids))

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",  # nosec B608
                memory_ids,
            )
            conn.commit()
            return cursor.rowcount

    def _build_query_conditions(self, filter: MemoryFilter) -> tuple[list[str], list[Any]]:
        """Build WHERE clause conditions from a MemoryFilter.

        Returns:
            Tuple of (conditions list, params list).
        """
        conditions: list[str] = []
        params: list[Any] = []

        # Hierarchical scope filtering
        if filter.user_id is not None:
            conditions.append("user_id = ?")
            params.append(filter.user_id)

            # Hierarchical filtering: when filtering by user_id only,
            # return USER-level and below (all that user's memories)
            # This is implicit - we just filter by user_id

            if filter.session_id is not None:
                conditions.append("session_id = ?")
                params.append(filter.session_id)

                # agent_id and turn_id are independent narrowing constraints:
                # turn_id must be applied even when agent_id is absent. Nesting
                # the turn_id check inside the agent_id block dropped the turn
                # filter for a (session_id + turn_id, no agent_id) query, so it
                # returned the whole session instead of the one turn.
                if filter.agent_id is not None:
                    conditions.append("agent_id = ?")
                    params.append(filter.agent_id)

                if filter.turn_id is not None:
                    conditions.append("turn_id = ?")
                    params.append(filter.turn_id)
            elif filter.agent_id is not None:
                # Agent without session - unusual but supported
                conditions.append("agent_id = ?")
                params.append(filter.agent_id)
            elif filter.turn_id is not None:
                # Turn without session/agent - unusual but supported
                conditions.append("turn_id = ?")
                params.append(filter.turn_id)
        elif filter.session_id is not None:
            # Session without user - filter by session only
            conditions.append("session_id = ?")
            params.append(filter.session_id)
        elif filter.agent_id is not None:
            # Agent without user/session
            conditions.append("agent_id = ?")
            params.append(filter.agent_id)
        elif filter.turn_id is not None:
            # Turn only
            conditions.append("turn_id = ?")
            params.append(filter.turn_id)

        # Explicit scope level filtering
        if filter.scope_levels is not None and len(filter.scope_levels) > 0:
            scope_conditions = []
            for level in filter.scope_levels:
                if level == ScopeLevel.USER:
                    # USER level: no session/agent/turn
                    scope_conditions.append(
                        "(session_id IS NULL AND agent_id IS NULL AND turn_id IS NULL)"
                    )
                elif level == ScopeLevel.SESSION:
                    # SESSION level: has session, no agent/turn
                    scope_conditions.append(
                        "(session_id IS NOT NULL AND agent_id IS NULL AND turn_id IS NULL)"
                    )
                elif level == ScopeLevel.AGENT:
                    # AGENT level: has agent, no turn
                    scope_conditions.append("(agent_id IS NOT NULL AND turn_id IS NULL)")
                elif level == ScopeLevel.TURN:
                    # TURN level: has turn
                    scope_conditions.append("(turn_id IS NOT NULL)")

            if scope_conditions:
                conditions.append(f"({' OR '.join(scope_conditions)})")

        # Temporal filtering
        if filter.created_after is not None:
            conditions.append("created_at >= ?")
            params.append(filter.created_after.isoformat())

        if filter.created_before is not None:
            conditions.append("created_at <= ?")
            params.append(filter.created_before.isoformat())

        # Point-in-time query
        if filter.valid_at is not None:
            valid_at_str = filter.valid_at.isoformat()
            conditions.append("valid_from <= ?")
            params.append(valid_at_str)
            conditions.append("(valid_until IS NULL OR valid_until > ?)")
            params.append(valid_at_str)

        # Superseded filtering. Skipped for point-in-time queries: valid_at
        # already constrains to versions valid at that instant, and ANDing
        # "valid_until IS NULL" on top would contradict it for any superseded
        # row — "what did we know at time t?" would always return nothing.
        if not filter.include_superseded and filter.valid_at is None:
            # Default: only return current memories (not superseded)
            conditions.append("valid_until IS NULL")

        # Importance filtering
        if filter.min_importance is not None:
            conditions.append("importance >= ?")
            params.append(filter.min_importance)

        if filter.max_importance is not None:
            conditions.append("importance <= ?")
            params.append(filter.max_importance)

        # Entity reference filtering (any of the specified entities)
        if filter.entity_refs is not None and len(filter.entity_refs) > 0:
            entity_conditions = []
            for entity_ref in filter.entity_refs:
                # Use JSON contains check
                entity_conditions.append("entity_refs LIKE ?")
                params.append(f'%"{entity_ref}"%')
            conditions.append(f"({' OR '.join(entity_conditions)})")

        # Lineage filtering
        if filter.has_supersedes is not None:
            if filter.has_supersedes:
                conditions.append("supersedes IS NOT NULL")
            else:
                conditions.append("supersedes IS NULL")

        if filter.has_promoted_from is not None:
            if filter.has_promoted_from:
                conditions.append("promoted_from IS NOT NULL")
            else:
                conditions.append("promoted_from IS NULL")

        # Metadata filtering with key validation to prevent JSON path injection
        if filter.metadata_filters:
            budget = [_MAX_METADATA_PREDICATES]
            for key, value in filter.metadata_filters.items():
                # Validate key to prevent JSON path injection attacks
                # Invalid keys are silently skipped to avoid breaking legitimate queries
                # while blocking malicious attempts like "'] OR 1=1--"
                if not _validate_metadata_key(key):
                    continue
                self._append_metadata_condition(f"$.{key}", value, conditions, params, budget)

        return conditions, params

    @staticmethod
    def _entry_mismatch_sql(s: str, f: str, depth: int) -> str:
        """SQL fragment: true when json_each rows ``s`` and ``f`` differ.

        Integer and real are grouped as one numeric class (matching the
        per-entry scalar branch, so results don't flip with container size).
        Containers recurse ``depth`` more structural join levels — each level
        keys objects by entry NAME (order-insensitive) and arrays by index
        (order enforced) — then compare as minified text below that. The
        nested json_each calls are guarded by CASE, which SQLite evaluates
        lazily, so they never run against a scalar value.
        """
        numeric = "('integer', 'real')"
        type_mismatch = (
            f"({s}.type != {f}.type AND NOT ({s}.type IN {numeric} AND {f}.type IN {numeric}))"
        )
        scalar_mismatch = (
            f"({f}.type IN ('integer', 'real', 'text') AND {s}.value IS NOT {f}.value)"
        )
        if depth <= 0:
            inner = f"{s}.value != {f}.value"
        else:
            s2, f2 = f"{s}x", f"{f}x"
            inner = (
                f"(SELECT COUNT(*) FROM json_each({s}.value))"
                f" != (SELECT COUNT(*) FROM json_each({f}.value))"
                f" OR EXISTS (SELECT 1 FROM json_each({f}.value) AS {f2}"
                f" LEFT JOIN json_each({s}.value) AS {s2} ON {s2}.key = {f2}.key"
                f" WHERE {s2}.key IS NULL"
                f" OR {SQLiteMemoryStore._entry_mismatch_sql(s2, f2, depth - 1)})"
            )
        container_mismatch = (
            f"({f}.type IN ('object', 'array')"
            f" AND CASE WHEN {s}.type = {f}.type THEN ({inner}) ELSE 1 END)"
        )
        return f"{type_mismatch} OR {scalar_mismatch} OR {container_mismatch}"

    def _bounded_container_equality(
        self, json_path: str, value: Any, conditions: list[str], params: list[Any]
    ) -> None:
        """Constant-size structural equality for a container filter value.

        Used when per-entry predicates would grow the SQL expression tree
        beyond SQLite's MAX_EXPR_DEPTH (oversized containers, spent predicate
        budget, or nested keys unsafe for JSON paths). Joins the filter's
        entries against the stored container's entries via json_each — on key
        for objects (key order stays irrelevant even for huge objects) and on
        index for arrays (order stays enforced) — and one more structural
        join level below that, so objects nested directly inside the
        container also match key-order-insensitively. Only containers three
        levels down compare as minified text (the remaining bounded
        compromise). Keys are joined as DATA, not interpolated into paths,
        so arbitrary key names are safe here. The SQL is constant-size
        regardless of container size.
        """
        json_kind = "array" if isinstance(value, (list, tuple)) else "object"
        serialized = json.dumps(value, separators=(",", ":"))
        conditions.append(f"json_type(metadata, '{json_path}') = '{json_kind}'")
        conditions.append(f"(SELECT COUNT(*) FROM json_each(metadata, '{json_path}')) = ?")
        params.append(len(value))
        conditions.append(
            "NOT EXISTS ("
            " SELECT 1 FROM json_each(?) AS f"
            f" LEFT JOIN json_each(metadata, '{json_path}') AS s ON s.key = f.key"
            " WHERE s.key IS NULL"
            f" OR {self._entry_mismatch_sql('s', 'f', 1)}"
            ")"
        )
        params.append(serialized)

    def _append_metadata_condition(
        self,
        json_path: str,
        value: Any,
        conditions: list[str],
        params: list[Any],
        budget: list[int],
    ) -> None:
        """Append typed equality conditions for one metadata JSON path.

        json_extract() returns typed SQL values (INTEGER for ints and
        booleans, REAL for floats, TEXT for strings AND for containers'
        minified JSON), so every branch pins both the comparable value and
        the JSON type — plain equality either silently matched nothing
        (JSON text 'true' vs integer 1) or matched across types (a string
        holding '["a"]' vs a real array).

        ``json_path`` is built exclusively from _validate_metadata_key-vetted
        segments, so interpolating it is injection-safe. ``budget`` is the
        shared predicate allowance for the whole metadata_filters build:
        per-container limits alone compose multiplicatively across nesting,
        which could still blow SQLite's expression-depth cap.
        """
        budget[0] -= 2
        if value is None:
            # `= NULL` never matches in SQL, and json_extract() maps BOTH an
            # explicit JSON null and a missing key to SQL NULL — an IS NULL
            # filter would return every memory that merely omits the key.
            # json_type() names an explicit null 'null' and returns SQL NULL
            # for a missing path, so equality matches only explicit nulls.
            conditions.append(f"json_type(metadata, '{json_path}') = 'null'")
            return
        if isinstance(value, bool):
            # json_extract() collapses JSON true/1 (and false/0) to the same
            # SQL integer. json_type() names booleans 'true'/'false' directly
            # — the type IS the value.
            conditions.append(f"json_type(metadata, '{json_path}') = ?")
            params.append("true" if value else "false")
            return
        if isinstance(value, (int, float)):
            # Require a numeric JSON type so {"active": true} doesn't match a
            # filter of 1 (the mirror of the boolean case).
            conditions.append(f"json_extract(metadata, '{json_path}') = ?")
            conditions.append(f"json_type(metadata, '{json_path}') IN ('integer', 'real')")
            params.append(value)
            return
        if isinstance(value, str):
            # Require the text type: json_extract exposes arrays and objects
            # as SQL text too, so a string filter holding minified JSON
            # (e.g. '["a"]') would otherwise also match a real container.
            conditions.append(f"json_extract(metadata, '{json_path}') = ?")
            conditions.append(f"json_type(metadata, '{json_path}') = 'text'")
            params.append(value)
            return
        if isinstance(value, dict):
            # JSON objects are unordered, but json_extract() preserves the
            # STORED key order while json.dumps preserves the FILTER's — raw
            # minified-text equality would miss logically equal objects.
            # Compare structurally: object type, entry count, and each entry
            # recursively. Oversized objects, spent budget, or nested keys
            # that can't be expressed as safe JSON paths all use the bounded
            # json_each-join comparison instead — which keys on entry NAME,
            # so it stays key-order-insensitive at this level and handles
            # arbitrary key spellings safely.
            if (
                len(value) > _MAX_STRUCTURAL_ITEMS
                or budget[0] <= 0
                or not all(_validate_metadata_key(k) for k in value)
            ):
                self._bounded_container_equality(json_path, value, conditions, params)
                return
            conditions.append(f"json_type(metadata, '{json_path}') = 'object'")
            conditions.append(f"(SELECT COUNT(*) FROM json_each(metadata, '{json_path}')) = ?")
            params.append(len(value))
            for sub_key, sub_value in value.items():
                self._append_metadata_condition(
                    f"{json_path}.{sub_key}", sub_value, conditions, params, budget
                )
            return
        # Arrays: JSON arrays are ordered, so compare element-by-element in
        # order — but recurse per element, because an element may itself be
        # an OBJECT whose key order must not matter (raw minified-text
        # equality of the whole array would be order-sensitive there).
        # Oversized arrays or a spent budget use the bounded comparison,
        # which joins on index so ordering stays enforced.
        if len(value) > _MAX_STRUCTURAL_ITEMS or budget[0] <= 0:
            self._bounded_container_equality(json_path, value, conditions, params)
            return
        conditions.append(f"json_type(metadata, '{json_path}') = 'array'")
        conditions.append(f"json_array_length(metadata, '{json_path}') = ?")
        params.append(len(value))
        for index, item in enumerate(value):
            self._append_metadata_condition(
                f"{json_path}[{index}]", item, conditions, params, budget
            )

    async def query(self, filter: MemoryFilter) -> list[Memory]:
        """Query memories matching the given filter.

        Args:
            filter: Filter criteria for the query.

        Returns:
            List of matching memories.
        """
        conditions, params = self._build_query_conditions(filter)

        # Build WHERE clause
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Build ORDER BY clause
        order_column = filter.order_by
        if order_column not in (
            "created_at",
            "importance",
            "access_count",
            "last_accessed",
        ):
            order_column = "created_at"

        order_direction = "DESC" if filter.order_desc else "ASC"

        # Build full query
        query = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY {order_column} {order_direction}
        """

        # Add pagination using parameterized queries to prevent injection
        if filter.limit is not None:
            query += " LIMIT ?"
            params.append(filter.limit)

        if filter.offset > 0:
            # SQLite only accepts OFFSET as part of a LIMIT clause; an OFFSET
            # without a LIMIT is a syntax error. When the caller paginates with
            # an offset but no limit, use SQLite's unbounded ``LIMIT -1``.
            if filter.limit is None:
                query += " LIMIT -1"
            query += " OFFSET ?"
            params.append(filter.offset)

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_memory(row) for row in cursor]

    async def count(self, filter: MemoryFilter) -> int:
        """Count memories matching the given filter.

        Args:
            filter: Filter criteria for the count.

        Returns:
            Number of matching memories.
        """
        conditions, params = self._build_query_conditions(filter)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"SELECT COUNT(*) FROM memories WHERE {where_clause}"

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()[0]
            return int(result)

    async def supersede(
        self,
        old_memory_id: str,
        new_memory: Memory,
        supersede_time: datetime | None = None,
    ) -> Memory:
        """Supersede an existing memory with a new version.

        This creates a temporal chain: the old memory's valid_until is set,
        and the new memory's supersedes field points to the old one.

        Args:
            old_memory_id: ID of the memory to supersede.
            new_memory: The new memory that replaces it.
            supersede_time: When the supersession occurred (defaults to now).

        Returns:
            The saved new memory with lineage fields populated.

        Raises:
            ValueError: If the old memory is not found.
        """
        if supersede_time is None:
            supersede_time = datetime.now(timezone.utc).replace(tzinfo=None)

        # Get the old memory
        old_memory = await self.get(old_memory_id)
        if old_memory is None:
            raise ValueError(f"Memory with ID {old_memory_id} not found")

        # Update old memory's valid_until and superseded_by
        old_memory.valid_until = supersede_time
        old_memory.superseded_by = new_memory.id

        # Set up new memory's lineage
        new_memory.supersedes = old_memory_id
        new_memory.valid_from = supersede_time

        # Save both in a transaction
        with self._get_conn() as conn:
            # Update old memory
            conn.execute(
                """
                UPDATE memories
                SET valid_until = ?, superseded_by = ?
                WHERE id = ?
                """,
                (supersede_time.isoformat(), new_memory.id, old_memory_id),
            )

            # Insert new memory
            row = self._memory_to_row(new_memory)
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, content, user_id, session_id, agent_id, turn_id,
                    created_at, valid_from, valid_until,
                    category, importance,
                    supersedes, superseded_by, promoted_from, promotion_chain,
                    access_count, last_accessed,
                    entity_refs, embedding, metadata
                ) VALUES (
                    :id, :content, :user_id, :session_id, :agent_id, :turn_id,
                    :created_at, :valid_from, :valid_until,
                    :category, :importance,
                    :supersedes, :superseded_by, :promoted_from, :promotion_chain,
                    :access_count, :last_accessed,
                    :entity_refs, :embedding, :metadata
                )
                """,
                row,
            )
            conn.commit()

        return new_memory

    async def detach_supersession(
        self,
        old_memory_id: str,
        new_memory_id: str,
    ) -> tuple[Memory, Memory]:
        """Atomically detach one verified supersession edge.

        This is an explicit repair operation. It never infers identity from
        content or embedding similarity and leaves neighboring chain edges
        untouched.
        """
        if old_memory_id == new_memory_id:
            raise ValueError("A memory cannot supersede itself")

        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM memories WHERE id IN (?, ?)",
                (old_memory_id, new_memory_id),
            ).fetchall()
            memories = {row["id"]: self._row_to_memory(row) for row in rows}
            old_memory = memories.get(old_memory_id)
            new_memory = memories.get(new_memory_id)

            if old_memory is None:
                raise ValueError(f"Memory {old_memory_id} not found")
            if new_memory is None:
                raise ValueError(f"Memory {new_memory_id} not found")
            if old_memory.superseded_by != new_memory_id or new_memory.supersedes != old_memory_id:
                raise ValueError(
                    f"Memories {old_memory_id} and {new_memory_id} do not form "
                    "a reciprocal supersession edge"
                )

            conn.execute(
                "UPDATE memories SET valid_until = NULL, superseded_by = NULL WHERE id = ?",
                (old_memory_id,),
            )
            conn.execute(
                "UPDATE memories SET supersedes = NULL WHERE id = ?",
                (new_memory_id,),
            )

        old_memory.valid_until = None
        old_memory.superseded_by = None
        new_memory.supersedes = None
        return old_memory, new_memory

    async def get_history(
        self,
        memory_id: str,
        include_future: bool = False,
    ) -> list[Memory]:
        """Get the full history chain for a memory.

        Follows the supersedes/superseded_by chain to return all versions.

        Args:
            memory_id: ID of any memory in the chain.
            include_future: Whether to include memories that superseded this one.

        Returns:
            List of memories in temporal order (oldest first).
        """
        # Start with the given memory
        current = await self.get(memory_id)
        if current is None:
            return []

        history: list[Memory] = [current]

        # Follow chain backwards (supersedes)
        back_id = current.supersedes
        while back_id is not None:
            prev = await self.get(back_id)
            if prev is None:
                break
            history.insert(0, prev)  # Add to beginning
            back_id = prev.supersedes

        # Follow chain forwards (superseded_by) if requested
        if include_future:
            forward_id = current.superseded_by
            while forward_id is not None:
                next_mem = await self.get(forward_id)
                if next_mem is None:
                    break
                history.append(next_mem)
                forward_id = next_mem.superseded_by

        return history

    async def clear_scope(
        self,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> int:
        """Clear all memories at or below a scope level.

        Args:
            user_id: Required user scope.
            session_id: If provided, clear session and below.
            agent_id: If provided, clear agent and below.
            turn_id: If provided, clear only that turn.

        Returns:
            Number of memories deleted.
        """
        conditions = ["user_id = ?"]
        params: list[Any] = [user_id]

        if turn_id is not None:
            # Clear only specific turn
            conditions.append("turn_id = ?")
            params.append(turn_id)
        elif agent_id is not None:
            # Clear agent and its turns
            conditions.append("agent_id = ?")
            params.append(agent_id)
        elif session_id is not None:
            # Clear session and its agents/turns
            conditions.append("session_id = ?")
            params.append(session_id)
        # If only user_id, clear all user's memories

        where_clause = " AND ".join(conditions)

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE {where_clause}",  # nosec B608
                params,
            )
            conn.commit()
            return cursor.rowcount

    async def clear_all(self) -> int:
        """Clear all memories from the store.

        Returns:
            Number of memories deleted.
        """
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM memories")
            conn.commit()
            return cursor.rowcount

    def count_sync(self) -> int:
        """Synchronous count of all memories (for diagnostics).

        Returns:
            Total number of memories in the store.
        """
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            result = cursor.fetchone()[0]
            return int(result)
