from __future__ import annotations

from collections.abc import Sequence

from isla_memory.models import Memory


def build_augmented_prompt(
    user_message: str,
    relevant_memories: Sequence[Memory],
) -> str:
    memory_lines = (
        "\n".join(f"- {memory.content}" for memory in relevant_memories)
        if relevant_memories
        else "- None"
    )
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
