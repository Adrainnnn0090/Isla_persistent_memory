from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from isla_memory.embedding_client import EmbeddingClient
from isla_memory.memory_store import MemoryStore
from isla_memory.models import CandidateMemory, Memory, MemoryDecision, Message, MemoryType
from isla_memory.utils import contains_any, normalize_text, parse_datetime, stable_id, utc_now

VALID_MEMORY_TYPES: set[str] = {"preference", "fact", "goal", "constraint", "profile", "other"}
VALID_TOOL_NAMES: dict[str, str] = {
    "add_memory": "ADD",
    "update_memory": "UPDATE",
    "delete_memory": "DELETE",
    "noop": "NOOP",
}


class MemoryDecisionClient(Protocol):
    def decide(
        self,
        *,
        candidate: CandidateMemory,
        retrieved_memories: list[tuple[Memory, float]],
        current_user_message: Message | None,
        source_metadata: dict[str, Any],
    ) -> Mapping[str, Any]:
        ...


class MemoryUpdater:
    def __init__(
        self,
        store: MemoryStore,
        embedding_client: EmbeddingClient,
        min_confidence: float = 0.65,
        dedup_score: float = 0.90,
        update_score: float = 0.62,
        update_strategy: str = "rules",
        update_top_s: int = 5,
        decision_client: MemoryDecisionClient | None = None,
        decision_min_confidence: float = 0.65,
        decision_fallback: str = "rules",
    ) -> None:
        self.store = store
        self.embedding_client = embedding_client
        self.min_confidence = min_confidence
        self.dedup_score = dedup_score
        self.update_score = update_score
        self.update_strategy = update_strategy
        self.update_top_s = update_top_s
        self.decision_client = decision_client
        self.decision_min_confidence = decision_min_confidence
        self.decision_fallback = decision_fallback

    def update_memories(
        self,
        user_id: str,
        candidates: list[CandidateMemory],
        current_user_message: Message | None = None,
    ) -> list[MemoryDecision]:
        return [
            self._apply_candidate(user_id, candidate, current_user_message)
            for candidate in candidates
        ]

    def _apply_candidate(
        self,
        user_id: str,
        candidate: CandidateMemory,
        current_user_message: Message | None,
    ) -> MemoryDecision:
        if candidate.confidence < self.min_confidence:
            return MemoryDecision(
                action="NOOP",
                candidate=candidate,
                confidence=candidate.confidence,
                reason="candidate confidence below threshold",
            )

        if self.update_strategy == "llm_tool_call" and self.decision_client is not None:
            return self._apply_candidate_with_llm_decision(
                user_id=user_id,
                candidate=candidate,
                current_user_message=current_user_message,
            )

        return self._apply_candidate_with_rules(user_id, candidate)

    def _apply_candidate_with_llm_decision(
        self,
        user_id: str,
        candidate: CandidateMemory,
        current_user_message: Message | None,
    ) -> MemoryDecision:
        query_text = self._candidate_search_text(candidate)
        candidate_embedding = self.embedding_client.embed(query_text)
        similar = self.store.similarity_search(
            user_id,
            candidate_embedding,
            top_k=self.update_top_s,
        )

        try:
            tool_call = self.decision_client.decide(
                candidate=candidate,
                retrieved_memories=similar,
                current_user_message=current_user_message,
                source_metadata=candidate.metadata,
            )
            decision = self._decision_from_tool_call(tool_call, candidate)
            validation_error = self._validate_llm_decision(decision, similar)
            if validation_error:
                return self._fallback_decision(user_id, candidate, validation_error)
            return self._apply_validated_decision(user_id, candidate, decision, similar)
        except Exception as exc:
            return self._fallback_decision(user_id, candidate, f"llm decision failed: {exc}")

    def _fallback_decision(
        self,
        user_id: str,
        candidate: CandidateMemory,
        reason: str,
    ) -> MemoryDecision:
        if self.decision_fallback != "rules":
            return MemoryDecision(
                action="NOOP",
                candidate=candidate,
                confidence=candidate.confidence,
                reason=f"{reason}; fallback disabled",
            )
        decision = self._apply_candidate_with_rules(user_id, candidate)
        decision.reason = f"{reason}; fallback rules: {decision.reason}"
        return decision

    def _apply_candidate_with_rules(
        self,
        user_id: str,
        candidate: CandidateMemory,
    ) -> MemoryDecision:
        if candidate.is_delete_intent:
            return self._apply_delete_with_rules(user_id, candidate)

        candidate_embedding = self.embedding_client.embed(candidate.content)
        similar = self.store.similarity_search(user_id, candidate_embedding, top_k=5)
        top_memory, top_score = similar[0] if similar else (None, 0.0)

        if top_memory is None or top_score < self.update_score:
            source_time = self._source_datetime(candidate)
            memory = Memory(
                memory_id=stable_id("mem"),
                user_id=user_id,
                content=candidate.content,
                embedding=candidate_embedding,
                memory_type=candidate.memory_type,
                created_at=source_time or utc_now(),
                updated_at=source_time or utc_now(),
                source_message_id=candidate.source_message_id,
                metadata=self._metadata_for_new_memory(candidate),
            )
            self.store.add_memory(memory)
            return MemoryDecision(
                action="ADD",
                candidate=candidate,
                final_content=memory.content,
                confidence=candidate.confidence,
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
                confidence=candidate.confidence,
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
            confidence=candidate.confidence,
            reason=f"similar memory found with score {top_score:.3f}",
        )

    def _apply_delete_with_rules(self, user_id: str, candidate: CandidateMemory) -> MemoryDecision:
        delete_query = self._candidate_search_text(candidate)
        query_embedding = self.embedding_client.embed(delete_query)
        similar = self.store.similarity_search(user_id, query_embedding, top_k=5)
        top_memory, top_score = similar[0] if similar else (None, 0.0)

        if top_memory is None or top_score < self.update_score:
            return MemoryDecision(
                action="NOOP",
                candidate=candidate,
                confidence=candidate.confidence,
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
            confidence=candidate.confidence,
            reason=f"soft-deleted matching memory with score {top_score:.3f}",
        )

    def _apply_validated_decision(
        self,
        user_id: str,
        candidate: CandidateMemory,
        decision: MemoryDecision,
        similar: list[tuple[Memory, float]],
    ) -> MemoryDecision:
        if decision.action == "NOOP":
            return decision

        if decision.action == "ADD":
            final_content = decision.final_content or candidate.content
            metadata = self._metadata_for_new_memory(candidate)
            metadata.update(decision.metadata_patch)
            memory_type = self._coerce_memory_type(
                str(metadata.get("memory_type", candidate.memory_type))
            )
            source_time = self._source_datetime(candidate)
            memory = Memory(
                memory_id=stable_id("mem"),
                user_id=user_id,
                content=final_content,
                embedding=self.embedding_client.embed(final_content),
                memory_type=memory_type,
                created_at=source_time or utc_now(),
                updated_at=source_time or utc_now(),
                source_message_id=candidate.source_message_id,
                metadata=metadata,
            )
            self.store.add_memory(memory)
            decision.final_content = memory.content
            decision.metadata_patch = metadata
            return decision

        if decision.action == "DELETE":
            self.store.delete_memory(
                decision.target_memory_id or "",
                reason=decision.reason,
            )
            return decision

        if decision.action == "UPDATE":
            target = self.store.get_memory(decision.target_memory_id or "")
            if target is None:
                return self._fallback_decision(
                    user_id,
                    candidate,
                    "validated update target disappeared before apply",
                )
            similarity = self._score_for_target(target.memory_id, similar)
            metadata = self._metadata_for_update(target, candidate, similarity)
            metadata.update(decision.metadata_patch)
            final_content = decision.final_content or candidate.content
            self.store.update_memory(
                memory_id=target.memory_id,
                content=final_content,
                embedding=self.embedding_client.embed(final_content),
                metadata=metadata,
            )
            decision.final_content = final_content
            decision.metadata_patch = metadata
            return decision

        return self._fallback_decision(user_id, candidate, "unsupported decision action")

    def _decision_from_tool_call(
        self,
        tool_call: Mapping[str, Any],
        candidate: CandidateMemory,
    ) -> MemoryDecision:
        tool_name = str(
            tool_call.get("name")
            or tool_call.get("tool_name")
            or tool_call.get("function")
            or ""
        )
        action = VALID_TOOL_NAMES.get(tool_name, tool_name)
        raw_arguments = tool_call.get("arguments", tool_call.get("args", {}))
        arguments = self._parse_arguments(raw_arguments)
        metadata_patch = arguments.get("metadata_patch", {})
        if not isinstance(metadata_patch, dict):
            metadata_patch = {}
        if "memory_type" in arguments:
            metadata_patch["memory_type"] = arguments["memory_type"]

        target_memory_id = arguments.get("memory_id")
        return MemoryDecision(
            action=action,  # type: ignore[arg-type]
            candidate=candidate,
            target_memory_id=str(target_memory_id) if target_memory_id else None,
            final_content=str(arguments.get("content")).strip()
            if arguments.get("content") is not None
            else None,
            confidence=float(arguments.get("confidence", 0.0)),
            reason=str(arguments.get("reason", "")).strip(),
            metadata_patch=metadata_patch,
        )

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, str):
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        if isinstance(raw_arguments, Mapping):
            return dict(raw_arguments)
        return {}

    def _validate_llm_decision(
        self,
        decision: MemoryDecision,
        similar: list[tuple[Memory, float]],
    ) -> str | None:
        if decision.action not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
            return f"invalid action {decision.action}"

        if decision.confidence < self.decision_min_confidence:
            return "decision confidence below threshold"

        target_ids = {memory.memory_id for memory, _score in similar}
        if decision.action in {"UPDATE", "DELETE"}:
            if not decision.target_memory_id:
                return f"{decision.action} missing memory_id"
            if decision.target_memory_id not in target_ids:
                return f"{decision.action} target not in retrieved memories"

        if decision.action in {"ADD", "UPDATE"} and not decision.final_content:
            return f"{decision.action} missing final content"

        if decision.action in {"ADD", "NOOP"} and decision.target_memory_id:
            return f"{decision.action} should not specify target_memory_id"

        return None

    @staticmethod
    def _candidate_search_text(candidate: CandidateMemory) -> str:
        if candidate.is_delete_intent:
            return str(candidate.metadata.get("delete_query") or candidate.content)
        return candidate.content

    @staticmethod
    def _score_for_target(memory_id: str, similar: list[tuple[Memory, float]]) -> float:
        for memory, score in similar:
            if memory.memory_id == memory_id:
                return score
        return 0.0

    @staticmethod
    def _coerce_memory_type(value: str) -> MemoryType:
        if value in VALID_MEMORY_TYPES:
            return value  # type: ignore[return-value]
        return "other"

    @staticmethod
    def _source_datetime(candidate: CandidateMemory) -> datetime | None:
        raw_value = candidate.metadata.get("source_date")
        if not raw_value:
            return None
        value = str(raw_value)
        try:
            return parse_datetime(value)
        except ValueError:
            pass
        for date_format in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(value, date_format).replace(tzinfo=UTC)
            except ValueError:
                continue
        return None

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
