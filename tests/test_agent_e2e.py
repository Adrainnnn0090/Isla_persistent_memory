from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isla_memory.agent import MemoryAgent
from isla_memory.config import MemoryConfig


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


if __name__ == "__main__":
    unittest.main()
