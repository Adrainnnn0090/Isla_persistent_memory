from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from isla_memory.embedding_client import HashEmbeddingClient
from isla_memory.memory_retriever import MemoryRetriever
from isla_memory.memory_store import MemoryStore
from isla_memory.models import Memory


class _FixedEmbeddingClient:
    def __init__(self, query_embedding: list[float]) -> None:
        self.query_embedding = query_embedding

    def embed(self, text: str) -> list[float]:
        del text
        return self.query_embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


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

    def test_top_k_min_score_and_similarity_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = _FixedEmbeddingClient([1.0, 0.0, 0.0])
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            store.add_memory(
                Memory(
                    memory_id="mem_best",
                    user_id="u1",
                    content="最相关 memory",
                    embedding=[1.0, 0.0, 0.0],
                )
            )
            store.add_memory(
                Memory(
                    memory_id="mem_second",
                    user_id="u1",
                    content="次相关 memory",
                    embedding=[0.8, 0.6, 0.0],
                )
            )
            store.add_memory(
                Memory(
                    memory_id="mem_low",
                    user_id="u1",
                    content="低相关 memory",
                    embedding=[0.0, 1.0, 0.0],
                )
            )
            retriever = MemoryRetriever(
                store,
                embedding,
                min_score=0.75,
                recency_weight=0.0,
                sim_weight=1.0,
            )

            scored = retriever.retrieve_with_scores("u1", "query", top_k=5)
            top_one = retriever.retrieve("u1", "query", top_k=1)

            self.assertEqual([memory.memory_id for memory, _score in scored], [
                "mem_best",
                "mem_second",
            ])
            self.assertEqual([memory.memory_id for memory in top_one], ["mem_best"])

    def test_unrelated_query_returns_empty_when_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = _FixedEmbeddingClient([1.0, 0.0, 0.0])
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            store.add_memory(
                Memory(
                    memory_id="mem_unrelated",
                    user_id="u1",
                    content="无关 memory",
                    embedding=[0.0, 1.0, 0.0],
                )
            )
            retriever = MemoryRetriever(
                store,
                embedding,
                min_score=0.10,
                recency_weight=0.0,
                sim_weight=1.0,
            )

            self.assertEqual(retriever.retrieve("u1", "query", top_k=5), [])

    def test_retrieval_filters_invalid_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = HashEmbeddingClient()
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            store.add_memory(
                Memory(
                    memory_id="mem_1",
                    user_id="u1",
                    content="用户偏好用中文回答技术问题。",
                    embedding=embedding.embed("用户偏好用中文回答技术问题。"),
                    memory_type="preference",
                )
            )
            store.delete_memory("mem_1", reason="test")
            retriever = MemoryRetriever(store, embedding, min_score=0.0)

            self.assertEqual(
                retriever.retrieve("u1", "我偏好什么回答方式？", top_k=5),
                [],
            )

    def test_recency_decay_can_rank_newer_memory_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = HashEmbeddingClient()
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            shared_embedding = embedding.embed("用户偏好回答方式。")
            store.add_memory(
                Memory(
                    memory_id="mem_old",
                    user_id="u1",
                    content="旧记忆：用户偏好简短回答。",
                    embedding=shared_embedding,
                    memory_type="preference",
                    updated_at=datetime.now(UTC) - timedelta(days=120),
                )
            )
            store.add_memory(
                Memory(
                    memory_id="mem_new",
                    user_id="u1",
                    content="新记忆：用户偏好详细回答。",
                    embedding=shared_embedding,
                    memory_type="preference",
                    updated_at=datetime.now(UTC),
                )
            )
            retriever = MemoryRetriever(
                store,
                embedding,
                min_score=0.0,
                recency_half_life_days=30,
                recency_weight=0.3,
                sim_weight=0.7,
            )

            memories = retriever.retrieve("u1", "用户偏好回答方式。", top_k=2)

            self.assertEqual(memories[0].memory_id, "mem_new")


if __name__ == "__main__":
    unittest.main()
