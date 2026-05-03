from __future__ import annotations

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
                metadata={"topic": "preference"},
            )

            store.add_memory(memory)
            self.assertEqual(store.get_memory("mem_1").content, "用户喜欢咖啡。")

            updated = store.update_memory(
                "mem_1",
                "用户喜欢热咖啡。",
                [0.9, 0.1, 0.0],
                {"confidence": 0.9},
            )
            self.assertEqual(updated.content, "用户喜欢热咖啡。")
            self.assertEqual(len(store.list_memories("u1")), 1)

            store.delete_memory("mem_1", reason="test")
            self.assertEqual(store.list_memories("u1"), [])
            invalid = store.get_memory("mem_1")
            self.assertIsNotNone(invalid)
            self.assertIsNotNone(invalid.invalid_at)
            self.assertEqual(len(store.list_memories("u1", include_invalid=True)), 1)


if __name__ == "__main__":
    unittest.main()
