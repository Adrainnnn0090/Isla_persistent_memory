from __future__ import annotations

from typing import Any

from isla_memory.embedding_client import EmbeddingClient
from isla_memory.memory_store import MemoryStore
from isla_memory.models import CandidateMemory, Memory, MemoryDecision
from isla_memory.utils import contains_any, normalize_text, stable_id, utc_now


class MemoryUpdater:
    def __init__(
        self,
        store: MemoryStore,
        embedding_client: EmbeddingClient,
        min_confidence: float = 0.65,
        dedup_score: float = 0.90,
        update_score: float = 0.62,
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client
        self.min_confidence = min_confidence
        self.dedup_score = dedup_score
        self.update_score = update_score

    def update_memories(
        self,
        user_id: str,
        candidates: list[CandidateMemory],
    ) -> list[MemoryDecision]:
        return [self._apply_candidate(user_id, candidate) for candidate in candidates]

    def _apply_candidate(self, user_id: str, candidate: CandidateMemory) -> MemoryDecision:
        if candidate.confidence < self.min_confidence:
            return MemoryDecision(
                action="NOOP",
                candidate=candidate,
                reason="candidate confidence below threshold",
            )

        if candidate.is_delete_intent:
            return self._apply_delete(user_id, candidate)

        candidate_embedding = self.embedding_client.embed(candidate.content)
        similar = self.store.similarity_search(user_id, candidate_embedding, top_k=5)
        top_memory, top_score = similar[0] if similar else (None, 0.0)

        if top_memory is None or top_score < self.update_score:
            memory = Memory(
                memory_id=stable_id("mem"),
                user_id=user_id,
                content=candidate.content,
                embedding=candidate_embedding,
                source_message_id=candidate.source_message_id,
                metadata=self._metadata_for_new_memory(candidate),
            )
            self.store.add_memory(memory)
            return MemoryDecision(
                action="ADD",
                candidate=candidate,
                target_memory_id=memory.memory_id,
                final_content=memory.content,
                reason="no similar active memory above update threshold",
            )

        if top_score >= self.dedup_score and self._contents_equivalent(
            top_memory.content,
            candidate.content,
        ):
            return MemoryDecision(
                action="NOOP",
                candidate=candidate,
                target_memory_id=top_memory.memory_id,
                final_content=top_memory.content,
                reason="candidate duplicates an existing memory",
            )

        final_content = self._merge_content(top_memory, candidate)
        final_embedding = self.embedding_client.embed(final_content)
        updated_metadata = self._metadata_for_update(top_memory, candidate, top_score)
        self.store.update_memory(
            memory_id=top_memory.memory_id,
            content=final_content,
            embedding=final_embedding,
            metadata=updated_metadata,
        )
        return MemoryDecision(
            action="UPDATE",
            candidate=candidate,
            target_memory_id=top_memory.memory_id,
            final_content=final_content,
            reason=f"similar memory found with score {top_score:.3f}",
        )

    def _apply_delete(self, user_id: str, candidate: CandidateMemory) -> MemoryDecision:
        delete_query = str(candidate.metadata.get("delete_query") or candidate.content)
        query_embedding = self.embedding_client.embed(delete_query)
        similar = self.store.similarity_search(user_id, query_embedding, top_k=5)
        top_memory, top_score = similar[0] if similar else (None, 0.0)

        if top_memory is None or top_score < self.update_score:
            return MemoryDecision(
                action="NOOP",
                candidate=candidate,
                reason="no active memory matched delete intent",
            )

        self.store.delete_memory(
            top_memory.memory_id,
            reason=f"delete intent matched with score {top_score:.3f}",
        )
        return MemoryDecision(
            action="DELETE",
            candidate=candidate,
            target_memory_id=top_memory.memory_id,
            final_content=top_memory.content,
            reason=f"soft-deleted matching memory with score {top_score:.3f}",
        )

    @staticmethod
    def _metadata_for_new_memory(candidate: CandidateMemory) -> dict[str, Any]:
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "memory_type": candidate.memory_type,
                "confidence": candidate.confidence,
                "source_message_id": candidate.source_message_id,
            }
        )
        return metadata

    @staticmethod
    def _metadata_for_update(
        existing: Memory,
        candidate: CandidateMemory,
        similarity: float,
    ) -> dict[str, Any]:
        history = list(existing.metadata.get("update_history", []))
        history.append(
            {
                "previous_content": existing.content,
                "candidate_content": candidate.content,
                "candidate_source_message_id": candidate.source_message_id,
                "similarity": similarity,
                "updated_at": utc_now().isoformat(),
            }
        )
        metadata = dict(existing.metadata)
        metadata.update(candidate.metadata)
        metadata["memory_type"] = candidate.memory_type
        metadata["confidence"] = max(
            float(metadata.get("confidence", 0.0)),
            candidate.confidence,
        )
        metadata["source_message_id"] = candidate.source_message_id
        metadata["update_history"] = history
        return metadata

    @staticmethod
    def _contents_equivalent(existing_content: str, candidate_content: str) -> bool:
        existing = normalize_text(existing_content)
        candidate = normalize_text(candidate_content)
        return existing == candidate or existing in candidate or candidate in existing

    def _merge_content(self, existing: Memory, candidate: CandidateMemory) -> str:
        if self._contents_equivalent(existing.content, candidate.content):
            longer = max((existing.content, candidate.content), key=len)
            return longer

        if self._is_communication_memory(existing, candidate):
            return self._merge_communication_content(existing.content, candidate.content)

        return f"{existing.content.rstrip('。')}；{candidate.content.rstrip('。')}。"

    @staticmethod
    def _is_communication_memory(existing: Memory, candidate: CandidateMemory) -> bool:
        if existing.metadata.get("topic") == "communication" or candidate.metadata.get("topic") == "communication":
            return True
        combined = f"{existing.content} {candidate.content}"
        return contains_any(combined, ("回答", "回复", "解释", "用中文", "用英文", "简洁", "直接"))

    @staticmethod
    def _merge_communication_content(existing_content: str, candidate_content: str) -> str:
        combined = f"{existing_content} {candidate_content}"
        style_parts: list[str] = []
        if contains_any(combined, ("中文", "汉语")):
            style_parts.append("用中文")
        if contains_any(combined, ("英文", "英语")):
            style_parts.append("用英文")
        if contains_any(combined, ("直接", "简洁", "短一点", "短些", "短")):
            style_parts.append("直接、简洁")
        if not style_parts:
            style_parts.append("按用户偏好的风格")

        scope = "技术问题" if contains_any(combined, ("技术", "代码", "编程")) else "问题"
        return f"用户偏好{'、'.join(style_parts)}地回答{scope}。"
