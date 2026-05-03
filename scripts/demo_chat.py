from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isla_memory.agent import MemoryAgent
from isla_memory.config import MemoryConfig


def print_turn(agent: MemoryAgent, message: str) -> None:
    print(f"User: {message}")
    print(f"Assistant: {agent.chat(message)}")
    if agent.last_decisions:
        for decision in agent.last_decisions:
            print(
                "Memory decision: "
                f"{decision.action} | {decision.final_content or decision.candidate.content}"
            )
    print()


def main() -> None:
    demo_db = ROOT / "data" / "demo_memory.sqlite3"
    if demo_db.exists():
        demo_db.unlink()

    config = MemoryConfig(db_path=str(demo_db))
    agent = MemoryAgent(user_id="demo_user", config=config)

    print_turn(agent, "以后请用中文回答技术问题，回答直接一点。")
    print_turn(agent, "我正在做一个类似 mem0 的简易长期记忆系统。")
    print_turn(agent, "我刚才告诉过你我的回答风格偏好吗？")

    print("Active memories:")
    for memory in agent.list_memories():
        print(f"- {memory.content}")

    print()
    print("Last augmented prompt:")
    print(agent.last_prompt)


if __name__ == "__main__":
    main()
