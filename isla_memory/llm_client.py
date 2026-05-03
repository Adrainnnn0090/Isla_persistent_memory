from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from isla_memory.models import Memory
from isla_memory.utils import contains_any


class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        user_message: str,
        relevant_memories: Sequence[Memory],
    ) -> str:
        ...


class RuleBasedLLMClient:
    """Small local responder for the runnable MVP demo."""

    def generate(
        self,
        prompt: str,
        user_message: str,
        relevant_memories: Sequence[Memory],
    ) -> str:
        del prompt
        question = user_message.strip()
        memory_text = "；".join(memory.content for memory in relevant_memories)

        if relevant_memories and contains_any(
            question,
            ("偏好什么", "什么回答", "回答方式", "回答风格", "告诉过你", "刚才说"),
        ):
            return f"你告诉过我：{memory_text}"

        if contains_any(question, ("以后请", "请用", "回答要", "直接一点", "短一点")):
            return "好的，我会按这个偏好来回答。"

        if contains_any(question, ("正在做", "记忆系统", "mem0", "长期记忆")):
            return "明白，我会把这个项目背景作为后续上下文。"

        if relevant_memories:
            return f"结合已知记忆：{memory_text}。我会据此回答当前问题。"

        return "我明白了。"
