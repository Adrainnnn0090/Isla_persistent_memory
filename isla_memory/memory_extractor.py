from __future__ import annotations

import re

from isla_memory.models import CandidateMemory, Message
from isla_memory.utils import contains_any, normalize_text


class RuleBasedMemoryExtractor:
    def extract_memories(
        self,
        user_id: str,
        recent_messages: list[Message],
        current_user_message: Message,
        current_assistant_message: Message | None = None,
    ) -> list[CandidateMemory]:
        del user_id, recent_messages, current_assistant_message
        text = current_user_message.content.strip()
        if not text:
            return []

        candidates: list[CandidateMemory] = []
        source_id = current_user_message.message_id

        if self._is_delete_intent(text):
            target = self._extract_delete_target(text)
            return [
                CandidateMemory(
                    content=f"用户希望作废相关记忆：{target}",
                    memory_type="constraint",
                    confidence=0.9,
                    source_message_id=source_id,
                    metadata={"intent": "delete", "delete_query": target, "raw_text": text},
                )
            ]

        if self._looks_like_question(text):
            return []

        communication_memory = self._extract_communication_preference(text, source_id)
        if communication_memory:
            candidates.append(communication_memory)

        project_memory = self._extract_project_context(text, source_id)
        if project_memory:
            candidates.append(project_memory)

        profile_memory = self._extract_profile(text, source_id)
        if profile_memory:
            candidates.append(profile_memory)

        like_memory = self._extract_generic_like(text, source_id)
        if like_memory:
            candidates.append(like_memory)

        goal_memory = self._extract_goal(text, source_id)
        if goal_memory:
            candidates.append(goal_memory)

        return self._dedupe(candidates)

    @staticmethod
    def _is_delete_intent(text: str) -> bool:
        return contains_any(
            text,
            (
                "不要再记住",
                "别再记住",
                "忘记我",
                "忘掉我",
                "不再记住",
                "取消记忆",
                "删除记忆",
                "delete this memory",
                "forget that",
            ),
        )

    @staticmethod
    def _extract_delete_target(text: str) -> str:
        patterns = (
            r"(?:不要再记住|别再记住|不再记住|忘记|忘掉|删除记忆|取消记忆)(.*)",
            r"(?:forget|delete)(.*)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.group(1).strip():
                return match.group(1).strip(" ：:，。.!！?")
        return text

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        question_markers = ("?", "？", "什么", "哪些", "怎么", "如何", "吗", "是不是")
        explicit_memory_markers = ("以后请", "请记住", "记住我", "我喜欢", "我偏好", "我的目标是")
        return contains_any(text, question_markers) and not contains_any(text, explicit_memory_markers)

    @staticmethod
    def _extract_communication_preference(
        text: str,
        source_message_id: str,
    ) -> CandidateMemory | None:
        if not contains_any(
            text,
            ("回答", "解释", "回复", "用中文", "用英文", "直接一点", "短一点", "简洁", "技术问题"),
        ):
            return None
        if not contains_any(text, ("以后", "请", "偏好", "喜欢", "希望", "回答要", "解释")):
            return None

        style_parts: list[str] = []
        if contains_any(text, ("中文", "汉语")):
            style_parts.append("用中文")
        if contains_any(text, ("英文", "英语")):
            style_parts.append("用英文")
        if contains_any(text, ("直接", "简洁", "短一点", "短些", "短", "直接一点")):
            style_parts.append("直接、简洁")

        scope = "技术问题" if contains_any(text, ("技术", "代码", "编程")) else "问题"
        if not style_parts:
            style_parts.append("按用户偏好的风格")

        return CandidateMemory(
            content=f"用户偏好{'、'.join(style_parts)}地回答{scope}。",
            memory_type="preference",
            confidence=0.93,
            source_message_id=source_message_id,
            metadata={"topic": "communication"},
        )

    @staticmethod
    def _extract_project_context(text: str, source_message_id: str) -> CandidateMemory | None:
        if contains_any(text, ("mem0", "记忆系统", "长期记忆系统")) and contains_any(
            text,
            ("做", "开发", "构建", "实现", "项目"),
        ):
            return CandidateMemory(
                content="用户正在做一个类似 mem0 的简易长期记忆系统。",
                memory_type="goal",
                confidence=0.93,
                source_message_id=source_message_id,
                metadata={"topic": "project", "project_type": "memory_system"},
            )

        match = re.search(r"我(?:现在)?(?:正在|在)?做(?:一个)?([^。！？!?]{2,40})", text)
        if match:
            description = match.group(1).strip(" ：:，,")
            if description:
                return CandidateMemory(
                    content=f"用户正在做{description}。",
                    memory_type="goal",
                    confidence=0.78,
                    source_message_id=source_message_id,
                    metadata={"topic": "project"},
                )
        return None

    @staticmethod
    def _extract_profile(text: str, source_message_id: str) -> CandidateMemory | None:
        match = re.search(r"(?:我叫|我的名字是)([\u4e00-\u9fffA-Za-z0-9_\-]{1,32})", text)
        if not match:
            return None
        return CandidateMemory(
            content=f"用户的名字是{match.group(1)}。",
            memory_type="profile",
            confidence=0.88,
            source_message_id=source_message_id,
            metadata={"topic": "profile", "field": "name"},
        )

    @staticmethod
    def _extract_generic_like(text: str, source_message_id: str) -> CandidateMemory | None:
        match = re.search(r"我(?:很)?喜欢([^。！？!?，,]{1,30})", text)
        if not match:
            return None

        target = match.group(1).strip(" ：:，,")
        if not target or contains_any(target, ("回答", "解释", "以后", "你以后")):
            return None

        return CandidateMemory(
            content=f"用户喜欢{target}。",
            memory_type="preference",
            confidence=0.82,
            source_message_id=source_message_id,
            metadata={"topic": "preference"},
        )

    @staticmethod
    def _extract_goal(text: str, source_message_id: str) -> CandidateMemory | None:
        match = re.search(r"我的目标是([^。！？!?]{2,60})", text)
        if not match:
            return None
        return CandidateMemory(
            content=f"用户的目标是{match.group(1).strip()}。",
            memory_type="goal",
            confidence=0.86,
            source_message_id=source_message_id,
            metadata={"topic": "goal"},
        )

    @staticmethod
    def _dedupe(candidates: list[CandidateMemory]) -> list[CandidateMemory]:
        seen: set[str] = set()
        result: list[CandidateMemory] = []
        for candidate in candidates:
            key = normalize_text(candidate.content)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result


def extract_memories(
    user_id: str,
    recent_messages: list[Message],
    current_user_message: Message,
    current_assistant_message: Message | None = None,
) -> list[CandidateMemory]:
    return RuleBasedMemoryExtractor().extract_memories(
        user_id=user_id,
        recent_messages=recent_messages,
        current_user_message=current_user_message,
        current_assistant_message=current_assistant_message,
    )
