from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from isla_memory.agent import MemoryAgent
from isla_memory.config import MemoryConfig
from isla_memory.embedding_client import BGEEmbeddingClient, HashEmbeddingClient
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

    def test_bge_embedding_client_uses_dense_vectors_in_batch_order(self) -> None:
        class _FakeBGEM3FlagModel:
            def __init__(self, model: str, use_fp16: bool, device: str | None) -> None:
                self.model = model
                self.use_fp16 = use_fp16
                self.device = device

            def encode(self, texts: list[str], **_kwargs: object) -> dict[str, list[list[float]]]:
                return {
                    "dense_vecs": [
                        [float(index), float(len(text))]
                        for index, text in enumerate(texts)
                    ]
                }

        fake_module = types.SimpleNamespace(BGEM3FlagModel=_FakeBGEM3FlagModel)
        with patch.dict(sys.modules, {"FlagEmbedding": fake_module}):
            client = BGEEmbeddingClient(
                model="BAAI/bge-m3",
                device="cpu",
                batch_size=2,
                max_length=8192,
                use_fp16=False,
            )

            vectors = client.embed_many(["alpha", "beta"])

        self.assertEqual(vectors, [[0.0, 5.0], [1.0, 4.0]])

    def test_bge_provider_can_be_built_from_config(self) -> None:
        class _FakeBGEM3FlagModel:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def encode(self, texts: list[str], **_kwargs: object) -> dict[str, list[list[float]]]:
                return {"dense_vecs": [[1.0, 0.0] for _text in texts]}

        fake_module = types.SimpleNamespace(BGEM3FlagModel=_FakeBGEM3FlagModel)
        config = MemoryConfig(
            embedding_provider="bge_m3",
            bge_model="BAAI/bge-m3",
            bge_device="cpu",
            bge_batch_size=2,
            bge_max_length=8192,
            bge_use_fp16=False,
        )

        with patch.dict(sys.modules, {"FlagEmbedding": fake_module}):
            client = MemoryAgent._build_embedding_client(config)

        self.assertIsInstance(client, BGEEmbeddingClient)
        self.assertEqual(client.embed("hello"), [1.0, 0.0])

    def test_bge_missing_dependency_error_mentions_install_extra(self) -> None:
        with patch.dict(sys.modules, {"FlagEmbedding": None}):
            with self.assertRaisesRegex(RuntimeError, r'\.\[local-embedding\]'):
                BGEEmbeddingClient(device="cpu")


if __name__ == "__main__":
    unittest.main()
