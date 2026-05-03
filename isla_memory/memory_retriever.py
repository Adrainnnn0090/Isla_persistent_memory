from __future__ import annotations

from isla_memory.embedding_client import EmbeddingClient
from isla_memory.memory_store import MemoryStore
from isla_memory.models import Memory


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedding_client: EmbeddingClient,
        min_score: float = 0.35,
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client
        self.min_score = min_score

    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[Memory]:
        return [
            memory
            for memory, _score in self.retrieve_with_scores(
                user_id=user_id,
                query=query,
                top_k=top_k,
                min_score=min_score,
            )
        ]

    def retrieve_with_scores(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[tuple[Memory, float]]:
        threshold = self.min_score if min_score is None else min_score
        query_embedding = self.embedding_client.embed(query)
        results = self.store.similarity_search(user_id, query_embedding, top_k=top_k)
        return [(memory, score) for memory, score in results if score >= threshold]
