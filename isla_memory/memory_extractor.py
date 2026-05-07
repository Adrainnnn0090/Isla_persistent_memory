from __future__ import annotations

import json
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
                "我不想再用",
                "我改变主意了",
                "以后不用",
                "取消",
                "delete this memory",
                "forget that",
            ),
        )

    @staticmethod
    def _extract_delete_target(text: str) -> str:
        patterns = (
            r"(?:不要再记住|别再记住|不再记住|忘记|忘掉|删除记忆|取消记忆)(.*)",
            r"(?:我不想再用|我改变主意了|以后不用|取消)(.*)",
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


class OpenAIMemoryExtractor:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        fallback_extractor: RuleBasedMemoryExtractor | None = None,
        include_assistant_facts: bool = False,
    ) -> None:
        self.model = model
        self.fallback_extractor = fallback_extractor or RuleBasedMemoryExtractor()
        self.include_assistant_facts = include_assistant_facts
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI extractor requires the "openai" package. '
                'Install it with: pip install -e ".[openai]"'
            ) from exc
        self.client = OpenAI(api_key=api_key)

    def extract_memories(
        self,
        user_id: str,
        recent_messages: list[Message],
        current_user_message: Message,
        current_assistant_message: Message | None = None,
    ) -> list[CandidateMemory]:
        if (
            not self.include_assistant_facts
            and self.fallback_extractor._looks_like_question(current_user_message.content)
        ):
            return []

        prompt = self._build_prompt(
            recent_messages,
            current_user_message,
            current_assistant_message,
            include_assistant_facts=self.include_assistant_facts,
        )
        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            output_text = getattr(response, "output_text", "") or str(response)
            return self._parse_candidates(output_text, current_user_message.message_id)
        except Exception:
            return self.fallback_extractor.extract_memories(
                user_id=user_id,
                recent_messages=recent_messages,
                current_user_message=current_user_message,
                current_assistant_message=current_assistant_message,
            )

    @staticmethod
    def _build_prompt(
        recent_messages: list[Message],
        current_user_message: Message,
        current_assistant_message: Message | None,
        include_assistant_facts: bool = False,
    ) -> str:
        recent_payload = (
            recent_messages[-10:]
            if include_assistant_facts
            else [message for message in recent_messages[-10:] if message.role == "user"]
        )
        payload = {
            "recent_messages": [message.to_dict() for message in recent_payload],
            "current_user_message": current_user_message.to_dict(),
        }
        if include_assistant_facts and current_assistant_message is not None:
            payload["current_assistant_message"] = current_assistant_message.to_dict()
        assistant_policy = (
            "Extract stable, verifiable facts from both user and assistant messages. "
            "Assistant facts are allowed because benchmark answers may depend on information "
            "the assistant provided in prior sessions."
            if include_assistant_facts
            else "Extract memories only from user-authored messages.\n"
            "Never extract memories from assistant-generated content."
        )
        return f"""You extract long-term user memories from conversation.

{assistant_policy}
Only extract information that is likely to be useful in future conversations.
Do not extract one-off tasks, temporary context, or trivial statements.
If the current user message is asking what the assistant remembers, return an empty memories list.
Avoid sensitive personal data unless the user explicitly asks the assistant to remember it.
If the user negates, cancels, changes, or asks to forget a previous preference or fact, return a DELETE intent candidate instead of a new positive memory.
DELETE intent examples:
- "不要再记住我喜欢 X" -> metadata.intent = "delete"
- "忘记我的项目偏好" -> metadata.intent = "delete"
- "我不想再用英文回答了" -> metadata.intent = "delete"
- "我改变主意了，不要简短回答" -> metadata.intent = "delete"

Return JSON only with this schema:
{{
  "memories": [
    {{
      "content": "...",
      "memory_type": "preference|fact|goal|constraint|profile|other",
      "confidence": 0.0,
      "source_message_id": "...",
      "metadata": {{}}
    }}
  ]
}}

Conversation payload:
{json.dumps(payload, ensure_ascii=False)}
"""

    @staticmethod
    def _parse_candidates(output_text: str, default_source_message_id: str) -> list[CandidateMemory]:
        data = json.loads(OpenAIMemoryExtractor._extract_json(output_text))
        raw_memories = data.get("memories", [])
        if not isinstance(raw_memories, list):
            return []

        candidates: list[CandidateMemory] = []
        for item in raw_memories:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            memory_type = str(item.get("memory_type", "other"))
            if memory_type not in {"preference", "fact", "goal", "constraint", "profile", "other"}:
                memory_type = "other"
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            candidates.append(
                CandidateMemory(
                    content=content,
                    memory_type=memory_type,  # type: ignore[arg-type]
                    confidence=float(item.get("confidence", 0.0)),
                    source_message_id=item.get("source_message_id") or default_source_message_id,
                    metadata=metadata,
                )
            )
        return candidates

    @staticmethod
    def _extract_json(output_text: str) -> str:
        stripped = output_text.strip()
        if stripped.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
            if match:
                return match.group(1)
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            return match.group(0)
        return stripped
