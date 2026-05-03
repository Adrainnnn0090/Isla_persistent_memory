from __future__ import annotations

import unittest

from isla_memory.memory_extractor import OpenAIMemoryExtractor, extract_memories
from isla_memory.models import Message
from isla_memory.utils import stable_id


class MemoryExtractorTest(unittest.TestCase):
    def test_extracts_communication_preference(self) -> None:
        message = Message(
            message_id=stable_id("msg"),
            user_id="u1",
            role="user",
            content="以后请用中文回答技术问题，回答直接一点。",
        )

        candidates = extract_memories("u1", [], message)

        self.assertEqual(len(candidates), 1)
        self.assertIn("用中文", candidates[0].content)
        self.assertEqual(candidates[0].memory_type, "preference")

    def test_ignores_one_off_question(self) -> None:
        message = Message(
            message_id=stable_id("msg"),
            user_id="u1",
            role="user",
            content="今天上海天气怎么样？",
        )

        self.assertEqual(extract_memories("u1", [], message), [])

    def test_extracts_delete_intent(self) -> None:
        message = Message(
            message_id=stable_id("msg"),
            user_id="u1",
            role="user",
            content="不要再记住我喜欢咖啡。",
        )

        candidates = extract_memories("u1", [], message)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].is_delete_intent)

    def test_openai_prompt_excludes_assistant_content(self) -> None:
        user_message = Message(
            message_id=stable_id("msg"),
            user_id="u1",
            role="user",
            content="我正在做一个类似 mem0 的简易长期记忆系统。",
        )
        assistant_message = Message(
            message_id=stable_id("msg"),
            user_id="u1",
            role="assistant",
            content="这里是一大段 assistant 生成的建议，不应该被抽取成用户记忆。",
        )

        prompt = OpenAIMemoryExtractor._build_prompt([], user_message, assistant_message)

        self.assertIn("current_user_message", prompt)
        self.assertNotIn("assistant 生成的建议", prompt)


if __name__ == "__main__":
    unittest.main()
