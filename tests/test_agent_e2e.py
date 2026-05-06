from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isla_memory.agent import MemoryAgent
from isla_memory.config import MemoryConfig
from isla_memory.models import Memory
from isla_memory.prompts import build_augmented_prompt


class AgentE2ETest(unittest.TestCase):
    def test_agent_runs_full_memory_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MemoryConfig(db_path=str(Path(tmp_dir) / "memory.sqlite3"))
            agent = MemoryAgent(user_id="u1", config=config)

            agent.chat("以后请用中文回答技术问题，回答直接一点。")
            agent.chat("我正在做一个类似 mem0 的简易长期记忆系统。")
            response = agent.chat("我刚才告诉过你我的回答风格偏好吗？")

            memories = agent.list_memories()
            self.assertGreaterEqual(len(memories), 2)
            self.assertIn("用中文", response)
            self.assertIn("Relevant user memories", agent.last_prompt)
            self.assertIn("standing preference", agent.last_prompt)
            self.assertTrue(agent.last_retrieved_memories)

    def test_prompt_builder_respects_memory_token_budget(self) -> None:
        long_memory = Memory(
            memory_id="mem_1",
            user_id="u1",
            content="用户偏好" + "直接回答" * 80,
            embedding=[1.0, 0.0, 0.0],
        )
        second_memory = Memory(
            memory_id="mem_2",
            user_id="u1",
            content="第二条记忆不应该进入低预算 prompt。",
            embedding=[0.0, 1.0, 0.0],
        )

        prompt = build_augmented_prompt(
            "我偏好什么回答方式？",
            [long_memory, second_memory],
            max_memory_tokens=8,
        )

        self.assertIn("Relevant user memories", prompt)
        self.assertIn("...", prompt)
        self.assertNotIn("第二条记忆", prompt)


if __name__ == "__main__":
    unittest.main()
