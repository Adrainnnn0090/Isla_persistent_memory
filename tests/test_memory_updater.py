from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isla_memory.embedding_client import HashEmbeddingClient
from isla_memory.memory_extractor import extract_memories
from isla_memory.memory_store import MemoryStore
from isla_memory.memory_updater import MemoryUpdater
from isla_memory.models import CandidateMemory, Memory, Message
from isla_memory.utils import stable_id


class _StaticDecisionClient:
    def __init__(self, tool_call: dict[str, object]) -> None:
        self.tool_call = tool_call
        self.calls: list[dict[str, object]] = []

    def decide(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.tool_call


class MemoryUpdaterTest(unittest.TestCase):
    def test_add_noop_update_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = HashEmbeddingClient()
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            updater = MemoryUpdater(store, embedding, update_score=0.62)
            candidate = CandidateMemory(
                content="用户偏好用中文地回答技术问题。",
                memory_type="preference",
                confidence=0.95,
                metadata={"topic": "communication"},
            )

            decisions = updater.update_memories("u1", [candidate])
            self.assertEqual(decisions[0].action, "ADD")

            decisions = updater.update_memories("u1", [candidate])
            self.assertEqual(decisions[0].action, "NOOP")

            updated_candidate = CandidateMemory(
                content="用户偏好用中文、直接、简洁地回答技术问题。",
                memory_type="preference",
                confidence=0.95,
                metadata={"topic": "communication"},
            )
            decisions = updater.update_memories("u1", [updated_candidate])
            self.assertEqual(decisions[0].action, "UPDATE")
            self.assertIn("直接", store.list_memories("u1")[0].content)

            delete_message = Message(
                message_id=stable_id("msg"),
                user_id="u1",
                role="user",
                content="不要再记住我用中文回答技术问题的偏好。",
            )
            delete_candidate = extract_memories("u1", [], delete_message)[0]
            decisions = updater.update_memories("u1", [delete_candidate])
            self.assertEqual(decisions[0].action, "DELETE")
            self.assertEqual(store.list_memories("u1"), [])
            self.assertEqual(len(store.list_memories("u1", include_invalid=True)), 1)

    def test_openai_style_candidate_without_topic_still_merges_communication_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = HashEmbeddingClient()
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            updater = MemoryUpdater(store, embedding, update_score=0.30)

            first = CandidateMemory(
                content="用户偏好技术问题用中文直接回答。",
                memory_type="preference",
                confidence=0.95,
            )
            second = CandidateMemory(
                content="用户希望以后技术问题用中文直接回答，回答要简洁明了。",
                memory_type="preference",
                confidence=0.95,
            )

            updater.update_memories("u1", [first])
            decision = updater.update_memories("u1", [second])[0]
            memory = store.list_memories("u1")[0]

            self.assertEqual(decision.action, "UPDATE")
            self.assertEqual(memory.content, "用户偏好用中文、直接、简洁地回答技术问题。")

    def test_low_confidence_candidate_noops_before_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = HashEmbeddingClient()
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            decision_client = _StaticDecisionClient({"name": "add_memory", "arguments": {}})
            updater = MemoryUpdater(
                store,
                embedding,
                update_strategy="llm_tool_call",
                decision_client=decision_client,
            )
            candidate = CandidateMemory(
                content="用户偏好用中文回答技术问题。",
                memory_type="preference",
                confidence=0.10,
            )

            decision = updater.update_memories("u1", [candidate])[0]

            self.assertEqual(decision.action, "NOOP")
            self.assertEqual(decision_client.calls, [])
            self.assertEqual(store.list_memories("u1"), [])

    def test_llm_tool_call_add_is_validated_and_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = HashEmbeddingClient()
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            decision_client = _StaticDecisionClient(
                {
                    "name": "add_memory",
                    "arguments": {
                        "content": "用户正在做一个长期记忆系统。",
                        "memory_type": "goal",
                        "metadata_patch": {"topic": "project"},
                        "reason": "new long-term project context",
                        "confidence": 0.91,
                    },
                }
            )
            updater = MemoryUpdater(
                store,
                embedding,
                update_strategy="llm_tool_call",
                decision_client=decision_client,
            )
            candidate = CandidateMemory(
                content="用户正在做一个长期记忆系统。",
                memory_type="goal",
                confidence=0.95,
            )

            decision = updater.update_memories("u1", [candidate])[0]
            memories = store.list_memories("u1")

            self.assertEqual(decision.action, "ADD")
            self.assertIsNone(decision.target_memory_id)
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].memory_type, "goal")
            self.assertIn("长期记忆系统", memories[0].content)

    def test_llm_tool_call_noop_does_not_duplicate(self) -> None:
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
            decision_client = _StaticDecisionClient(
                {
                    "name": "noop",
                    "arguments": {
                        "reason": "duplicate preference",
                        "confidence": 0.93,
                    },
                }
            )
            updater = MemoryUpdater(
                store,
                embedding,
                update_strategy="llm_tool_call",
                decision_client=decision_client,
            )
            candidate = CandidateMemory(
                content="用户偏好用中文回答技术问题。",
                memory_type="preference",
                confidence=0.95,
            )

            decision = updater.update_memories("u1", [candidate])[0]

            self.assertEqual(decision.action, "NOOP")
            self.assertEqual(len(store.list_memories("u1")), 1)

    def test_llm_tool_call_update_is_validated_and_applied(self) -> None:
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
            decision_client = _StaticDecisionClient(
                {
                    "name": "update_memory",
                    "arguments": {
                        "memory_id": "mem_1",
                        "content": "用户偏好用中文、直接、简洁地回答技术问题。",
                        "metadata_patch": {"topic": "communication"},
                        "reason": "more specific preference",
                        "confidence": 0.94,
                    },
                }
            )
            updater = MemoryUpdater(
                store,
                embedding,
                update_strategy="llm_tool_call",
                decision_client=decision_client,
            )
            candidate = CandidateMemory(
                content="用户偏好用中文、直接、简洁地回答技术问题。",
                memory_type="preference",
                confidence=0.95,
            )

            decision = updater.update_memories("u1", [candidate])[0]
            memory = store.get_memory("mem_1")

            self.assertEqual(decision.action, "UPDATE")
            self.assertIsNotNone(memory)
            self.assertIn("简洁", memory.content)
            self.assertEqual(memory.memory_type, "preference")

    def test_llm_tool_call_delete_soft_deletes_target(self) -> None:
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
            decision_client = _StaticDecisionClient(
                {
                    "name": "delete_memory",
                    "arguments": {
                        "memory_id": "mem_1",
                        "reason": "user cancelled preference",
                        "confidence": 0.92,
                    },
                }
            )
            updater = MemoryUpdater(
                store,
                embedding,
                update_strategy="llm_tool_call",
                decision_client=decision_client,
            )
            candidate = CandidateMemory(
                content="用户希望作废相关记忆：用中文回答技术问题",
                memory_type="constraint",
                confidence=0.95,
                metadata={"intent": "delete", "delete_query": "用中文回答技术问题"},
            )

            decision = updater.update_memories("u1", [candidate])[0]

            self.assertEqual(decision.action, "DELETE")
            self.assertEqual(store.list_memories("u1"), [])
            self.assertIsNotNone(store.get_memory("mem_1").invalid_at)

    def test_invalid_llm_tool_call_falls_back_to_rules(self) -> None:
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
            decision_client = _StaticDecisionClient(
                {
                    "name": "delete_memory",
                    "arguments": {
                        "memory_id": "mem_not_retrieved",
                        "reason": "invalid target",
                        "confidence": 0.95,
                    },
                }
            )
            updater = MemoryUpdater(
                store,
                embedding,
                update_strategy="llm_tool_call",
                decision_client=decision_client,
            )
            candidate = CandidateMemory(
                content="用户偏好用中文回答技术问题。",
                memory_type="preference",
                confidence=0.95,
            )

            decision = updater.update_memories("u1", [candidate])[0]

            self.assertEqual(decision.action, "NOOP")
            self.assertIn("fallback rules", decision.reason)
            self.assertEqual(len(store.list_memories("u1")), 1)

    def test_add_uses_candidate_source_date_for_memory_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = HashEmbeddingClient()
            store = MemoryStore(str(Path(tmp_dir) / "memory.sqlite3"))
            updater = MemoryUpdater(store, embedding)
            candidate = CandidateMemory(
                content="用户喜欢研究长期记忆系统。",
                memory_type="preference",
                confidence=0.95,
                metadata={"source_date": "2023/05/20 (Sat) 02:21"},
            )

            updater.update_memories("u1", [candidate])
            memory = store.list_memories("u1")[0]

            self.assertEqual(memory.created_at.year, 2023)
            self.assertEqual(memory.created_at.month, 5)
            self.assertEqual(memory.created_at.day, 20)


if __name__ == "__main__":
    unittest.main()
