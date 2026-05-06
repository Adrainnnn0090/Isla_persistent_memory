from __future__ import annotations

import math
from datetime import UTC, datetime

from isla_memory.embedding_client import EmbeddingClient
from isla_memory.memory_store import MemoryStore
from isla_memory.models import Memory


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedding_client: EmbeddingClient,
        min_score: float = 0.35,
        recency_half_life_days: int = 30,
        recency_weight: float = 0.3,
        sim_weight: float = 0.7,
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client
        self.min_score = min_score
        self.recency_half_life_days = recency_half_life_days
        self.recency_weight = recency_weight
        self.sim_weight = sim_weight

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
        candidate_k = max(top_k, top_k * 4)
        results = self.store.similarity_search(user_id, query_embedding, top_k=candidate_k)
        scored = [
            (memory, self._score_with_recency(memory, similarity))
            for memory, similarity in results
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(memory, score) for memory, score in scored if score >= threshold][:top_k]

    def _score_with_recency(self, memory: Memory, similarity: float) -> float:
        if self.recency_weight <= 0:
            return similarity
        return (
            similarity * self.sim_weight
            + self._recency_score(memory) * self.recency_weight
        )

    def _recency_score(self, memory: Memory) -> float:
        if self.recency_half_life_days <= 0:
            return 0.0
        now = datetime.now(UTC)
        updated_at = memory.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        age_days = max((now - updated_at).total_seconds() / 86400, 0.0)
        return math.exp(-0.693 * age_days / self.recency_half_life_days)
