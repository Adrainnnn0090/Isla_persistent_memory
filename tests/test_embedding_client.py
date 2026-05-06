from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isla_memory.embedding_client import HashEmbeddingClient
from isla_memory.memory_store import MemoryStore
from isla_memory.models import Memory
from isla_memory.utils import cosine_similarity


class EmbeddingClientTest(unittest.TestCase):
    def test_hash_embedding_is_deterministic_and_self_similar(self) -> None:
        embedding = HashEmbeddingClient()
        text = "用户偏好用中文、直接、简洁地回答技术问题。"

        first = embedding.embed(text)
        second = embedding.embed(text)

        self.assertEqual(first, second)
        self.assertAlmostEqual(cosine_similarity(first, second), 1.0)

    def test_related_text_scores_above_unrelated_text(self) -> None:
        embedding = HashEmbeddingClient()
        query = embedding.embed("我偏好什么回答方式？")
        related = embedding.embed("用户偏好用中文、直接、简洁地回答技术问题。")
        unrelated = embedding.embed("用户喜欢热咖啡。")

        self.assertGreater(
            cosine_similarity(query, related),
            cosine_similarity(query, unrelated),
        )

    def test_store_similarity_search_ranks_and_filters_by_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            store.add_memory(
                Memory(
                    memory_id="mem_related",
                    user_id="u1",
                    content="相关 memory",
                    embedding=[1.0, 0.0, 0.0],
                )
            )
            store.add_memory(
                Memory(
                    memory_id="mem_unrelated",
                    user_id="u1",
                    content="无关 memory",
                    embedding=[0.0, 1.0, 0.0],
                )
            )
            store.add_memory(
                Memory(
                    memory_id="mem_other_user",
                    user_id="u2",
                    content="其他用户 memory",
                    embedding=[1.0, 0.0, 0.0],
                )
            )

            results = store.similarity_search("u1", [1.0, 0.0, 0.0], top_k=5)

            self.assertEqual([memory.memory_id for memory, _score in results], [
                "mem_related",
                "mem_unrelated",
            ])
            self.assertGreater(results[0][1], results[1][1])


if __name__ == "__main__":
    unittest.main()
