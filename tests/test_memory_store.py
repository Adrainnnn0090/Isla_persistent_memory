from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from isla_memory.memory_store import MemoryStore
from isla_memory.models import Memory


class MemoryStoreTest(unittest.TestCase):
    def test_crud_and_soft_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            memory = Memory(
                memory_id="mem_1",
                user_id="u1",
                content="用户喜欢咖啡。",
                embedding=[1.0, 0.0, 0.0],
                memory_type="preference",
                metadata={"topic": "preference"},
            )

            store.add_memory(memory)
            stored = store.get_memory("mem_1")
            self.assertEqual(stored.content, "用户喜欢咖啡。")
            self.assertEqual(stored.memory_type, "preference")

            updated = store.update_memory(
                "mem_1",
                "用户喜欢热咖啡。",
                [0.9, 0.1, 0.0],
                {"confidence": 0.9},
            )
            self.assertEqual(updated.content, "用户喜欢热咖啡。")
            self.assertEqual(updated.memory_type, "preference")
            self.assertEqual(len(store.list_memories("u1")), 1)

            store.delete_memory("mem_1", reason="test")
            self.assertEqual(store.list_memories("u1"), [])
            invalid = store.get_memory("mem_1")
            self.assertIsNotNone(invalid)
            self.assertIsNotNone(invalid.invalid_at)
            self.assertEqual(len(store.list_memories("u1", include_invalid=True)), 1)
            self.assertEqual(store.similarity_search("u1", [0.9, 0.1, 0.0]), [])
            self.assertEqual(
                len(store.similarity_search("u1", [0.9, 0.1, 0.0], include_invalid=True)),
                1,
            )

    def test_migrates_legacy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "memory.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE memories (
                        memory_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        source_message_id TEXT,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO memories (
                        memory_id,
                        user_id,
                        content,
                        embedding_json,
                        created_at,
                        updated_at,
                        source_message_id,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "mem_legacy",
                        "u1",
                        "旧 schema memory。",
                        "[1.0, 0.0, 0.0]",
                        "2026-05-06T00:00:00+00:00",
                        "2026-05-06T00:00:00+00:00",
                        None,
                        "{}",
                    ),
                )

            store = MemoryStore(str(db_path))
            migrated = store.get_memory("mem_legacy")

            self.assertIsNotNone(migrated)
            self.assertEqual(migrated.memory_type, "other")
            self.assertIsNone(migrated.invalid_at)


if __name__ == "__main__":
    unittest.main()
