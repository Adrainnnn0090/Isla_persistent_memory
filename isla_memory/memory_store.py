from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from isla_memory.models import Memory
from isla_memory.utils import cosine_similarity, datetime_to_iso, parse_datetime, utc_now


class MemoryStore:
    def __init__(self, db_path: str = "./data/memory.sqlite3") -> None:
        self.db_path = db_path
        path = Path(db_path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'other',
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_message_id TEXT,
                    metadata_json TEXT NOT NULL,
                    invalid_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            self._migrate_schema(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(user_id, invalid_at)"
            )

    def add_memory(self, memory: Memory) -> Memory:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id,
                    user_id,
                    content,
                    memory_type,
                    embedding_json,
                    created_at,
                    updated_at,
                    source_message_id,
                    metadata_json,
                    invalid_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._to_row(memory),
            )
        return memory

    def update_memory(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        existing = self.get_memory(memory_id)
        if existing is None:
            raise KeyError(f"Memory not found: {memory_id}")

        merged_metadata = dict(existing.metadata)
        if metadata:
            merged_metadata.update(metadata)

        updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET content = ?,
                    memory_type = ?,
                    embedding_json = ?,
                    metadata_json = ?,
                    updated_at = ?,
                    invalid_at = NULL
                WHERE memory_id = ?
                """,
                (
                    content,
                    str(merged_metadata.get("memory_type", existing.memory_type)),
                    json.dumps(embedding),
                    json.dumps(merged_metadata, ensure_ascii=False),
                    datetime_to_iso(updated_at),
                    memory_id,
                ),
            )

        updated = self.get_memory(memory_id)
        if updated is None:
            raise KeyError(f"Memory not found after update: {memory_id}")
        return updated

    def delete_memory(self, memory_id: str, reason: str = "") -> None:
        """Soft-delete a memory by marking it invalid.

        Invalid memories are kept for temporal reasoning and auditability, but
        retrieval ignores them by default.
        """

        existing = self.get_memory(memory_id)
        if existing is None:
            return

        invalid_at = utc_now()
        metadata = dict(existing.metadata)
        metadata["invalidated_at"] = datetime_to_iso(invalid_at)
        if reason:
            metadata["invalid_reason"] = reason

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET invalid_at = ?,
                    updated_at = ?,
                    metadata_json = ?
                WHERE memory_id = ?
                """,
                (
                    datetime_to_iso(invalid_at),
                    datetime_to_iso(invalid_at),
                    json.dumps(metadata, ensure_ascii=False),
                    memory_id,
                ),
            )

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def list_memories(self, user_id: str, include_invalid: bool = False) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE user_id = ?"
        params: tuple[Any, ...] = (user_id,)
        if not include_invalid:
            sql += " AND invalid_at IS NULL"
        sql += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def similarity_search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        include_invalid: bool = False,
    ) -> list[tuple[Memory, float]]:
        memories = self.list_memories(user_id, include_invalid=include_invalid)
        scored = [
            (memory, cosine_similarity(query_embedding, memory.embedding))
            for memory in memories
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        if "invalid_at" not in columns:
            conn.execute("ALTER TABLE memories ADD COLUMN invalid_at TEXT")
        if "memory_type" not in columns:
            conn.execute(
                "ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'other'"
            )

    @staticmethod
    def _to_row(memory: Memory) -> tuple[Any, ...]:
        return (
            memory.memory_id,
            memory.user_id,
            memory.content,
            memory.memory_type,
            json.dumps(memory.embedding),
            datetime_to_iso(memory.created_at),
            datetime_to_iso(memory.updated_at),
            memory.source_message_id,
            json.dumps(memory.metadata, ensure_ascii=False),
            datetime_to_iso(memory.invalid_at),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Memory:
        return Memory(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            embedding=json.loads(row["embedding_json"]),
            created_at=parse_datetime(row["created_at"]) or utc_now(),
            updated_at=parse_datetime(row["updated_at"]) or utc_now(),
            source_message_id=row["source_message_id"],
            metadata=json.loads(row["metadata_json"]),
            invalid_at=parse_datetime(row["invalid_at"]),
        )
