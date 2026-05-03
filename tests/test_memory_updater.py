from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isla_memory.embedding_client import HashEmbeddingClient
from isla_memory.memory_extractor import extract_memories
from isla_memory.memory_store import MemoryStore
from isla_memory.memory_updater import MemoryUpdater
from isla_memory.models import CandidateMemory, Message
from isla_memory.utils import stable_id


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


if __name__ == "__main__":
    unittest.main()
