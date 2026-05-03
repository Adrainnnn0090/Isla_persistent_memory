from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isla_memory.embedding_client import HashEmbeddingClient
from isla_memory.memory_retriever import MemoryRetriever
from isla_memory.memory_store import MemoryStore
from isla_memory.models import Memory


class MemoryRetrieverTest(unittest.TestCase):
    def test_retrieves_relevant_memory_for_user_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = HashEmbeddingClient()
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            store.add_memory(
                Memory(
                    memory_id="mem_1",
                    user_id="u1",
                    content="用户偏好用中文、直接、简洁地回答技术问题。",
                    embedding=embedding.embed("用户偏好用中文、直接、简洁地回答技术问题。"),
                )
            )
            store.add_memory(
                Memory(
                    memory_id="mem_2",
                    user_id="u2",
                    content="用户喜欢咖啡。",
                    embedding=embedding.embed("用户喜欢咖啡。"),
                )
            )
            retriever = MemoryRetriever(store, embedding, min_score=0.30)

            memories = retriever.retrieve("u1", "我偏好什么回答方式？", top_k=5)

            self.assertEqual(len(memories), 1)
            self.assertIn("用中文", memories[0].content)


if __name__ == "__main__":
    unittest.main()
