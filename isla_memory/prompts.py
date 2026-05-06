from __future__ import annotations

from collections.abc import Sequence

from isla_memory.models import Memory


def build_augmented_prompt(
    user_message: str,
    relevant_memories: Sequence[Memory],
    max_memory_tokens: int = 800,
) -> str:
    memory_lines = _format_memory_lines(relevant_memories, max_memory_tokens)
    return f"""You are a helpful assistant.

Relevant user memories:
{memory_lines}

Memory usage rules:
- If a memory describes the user's communication or answer style, follow it as a standing preference.
- Do not restate memories unless the user asks about them.
- If the current message is only a statement or status update, acknowledge briefly instead of giving a full tutorial.
- Keep the answer concise by default.

Current question:
{user_message}

Use relevant memories only when they are useful for answering the current question.
"""


def _format_memory_lines(
    relevant_memories: Sequence[Memory],
    max_memory_tokens: int,
) -> str:
    if not relevant_memories or max_memory_tokens <= 0:
        return "- None"

    lines: list[str] = []
    used_tokens = 0
    for memory in relevant_memories:
        line = f"- {memory.content}"
        estimated_tokens = _estimate_tokens(line)
        if lines and used_tokens + estimated_tokens > max_memory_tokens:
            break
        if not lines and estimated_tokens > max_memory_tokens:
            lines.append(_truncate_to_budget(line, max_memory_tokens))
            break
        lines.append(line)
        used_tokens += estimated_tokens

    return "\n".join(lines) if lines else "- None"


def _estimate_tokens(text: str) -> int:
    # A conservative approximation that works without model-specific tokenizers.
    return max(1, int(len(text) / 4) + 1)


def _truncate_to_budget(text: str, max_tokens: int) -> str:
    max_chars = max(max_tokens * 4, 1)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
