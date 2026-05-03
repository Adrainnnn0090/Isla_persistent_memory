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

Current question:
{user_message}

Use relevant memories only when they are useful for answering the current question.
"""
