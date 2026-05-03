from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isla_memory.config import MemoryConfig
from isla_memory.embedding_client import HashEmbeddingClient, OpenAIEmbeddingClient
from isla_memory.llm_client import OpenAILLMClient
from isla_memory.memory_retriever import MemoryRetriever
from isla_memory.memory_store import MemoryStore
from isla_memory.models import Memory
from isla_memory.utils import stable_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LongMemEval hypothesis jsonl with Isla memory retrieval.",
    )
    parser.add_argument("--data", required=True, help="Path to a LongMemEval json file.")
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved memories.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Similarity threshold. Use 0.0 for benchmark-style forced top-k.",
    )
    parser.add_argument(
        "--granularity",
        choices=("session", "turn"),
        default="session",
        help="Index whole sessions or individual turns.",
    )
    parser.add_argument(
        "--question-type",
        action="append",
        default=None,
        help="Filter to one or more question_type values. Can be repeated.",
    )
    parser.add_argument(
        "--db-path",
        default="./data/longmemeval_eval.sqlite3",
        help="Temporary SQLite path for the benchmark run.",
    )
    return parser.parse_args()


def build_embedding_client(config: MemoryConfig) -> HashEmbeddingClient | OpenAIEmbeddingClient:
    if config.embedding_provider == "openai":
        return OpenAIEmbeddingClient(
            model=config.embedding_model,
            api_key=config.openai_api_key,
        )
    return HashEmbeddingClient(dimension=config.hash_embedding_dimension)


def load_examples(path: str, limit: int | None, question_types: list[str] | None) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        examples = json.load(file)
    if question_types:
        allowed = set(question_types)
        examples = [item for item in examples if item.get("question_type") in allowed]
    if limit is not None:
        examples = examples[:limit]
    return examples


def index_example(
    store: MemoryStore,
    embedding_client: HashEmbeddingClient | OpenAIEmbeddingClient,
    example: dict[str, Any],
    granularity: str,
) -> str:
    user_id = str(example["question_id"])
    sessions = example.get("haystack_sessions", [])
    session_ids = example.get("haystack_session_ids", [])
    dates = example.get("haystack_dates", [])

    for session_index, session in enumerate(sessions):
        session_id = str(session_ids[session_index]) if session_index < len(session_ids) else str(session_index)
        date = str(dates[session_index]) if session_index < len(dates) else ""

        if granularity == "session":
            content = format_session(session_id, date, session)
            add_memory(store, embedding_client, user_id, content, session_id, date, "session")
            continue

        for turn_index, turn in enumerate(session):
            content = format_turn(session_id, date, turn_index, turn)
            add_memory(store, embedding_client, user_id, content, session_id, date, "turn")

    return user_id


def add_memory(
    store: MemoryStore,
    embedding_client: HashEmbeddingClient | OpenAIEmbeddingClient,
    user_id: str,
    content: str,
    session_id: str,
    date: str,
    granularity: str,
) -> None:
    store.add_memory(
        Memory(
            memory_id=stable_id("lme"),
            user_id=user_id,
            content=content,
            embedding=embedding_client.embed(content),
            metadata={
                "benchmark": "longmemeval",
                "session_id": session_id,
                "date": date,
                "granularity": granularity,
            },
        )
    )


def format_session(session_id: str, date: str, session: list[dict[str, Any]]) -> str:
    lines = [f"[session_id: {session_id}]", f"[date: {date}]"]
    for turn_index, turn in enumerate(session):
        role = str(turn.get("role", "unknown")).capitalize()
        content = str(turn.get("content", "")).strip()
        lines.append(f"{turn_index + 1}. {role}: {content}")
    return "\n".join(lines)


def format_turn(session_id: str, date: str, turn_index: int, turn: dict[str, Any]) -> str:
    role = str(turn.get("role", "unknown")).capitalize()
    content = str(turn.get("content", "")).strip()
    return "\n".join(
        [
            f"[session_id: {session_id}]",
            f"[date: {date}]",
            f"[turn_index: {turn_index}]",
            f"{role}: {content}",
        ]
    )


def build_answer_prompt(example: dict[str, Any], retrieved: list[Memory]) -> str:
    memory_text = "\n\n".join(
        f"Memory {index + 1}:\n{memory.content}"
        for index, memory in enumerate(retrieved)
    )
    if not memory_text:
        memory_text = "No relevant memory retrieved."

    return f"""You answer LongMemEval questions using only the retrieved chat history.

Question date:
{example.get("question_date", "")}

Retrieved chat history:
{memory_text}

Question:
{example["question"]}

Answer concisely. If the retrieved history does not contain enough evidence, say you do not know based on the provided history.
"""


def main() -> None:
    args = parse_args()
    config = MemoryConfig.from_env()
    if config.llm_provider != "openai":
        raise SystemExit("Set MEMORY_LLM_PROVIDER=openai before running LongMemEval generation.")

    db_path = Path(args.db_path)
    if db_path.exists():
        db_path.unlink()

    examples = load_examples(args.data, args.limit, args.question_type)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(str(db_path))
    embedding_client = build_embedding_client(config)
    retriever = MemoryRetriever(store, embedding_client, min_score=args.min_score)
    llm_client = OpenAILLMClient(model=config.llm_model, api_key=config.openai_api_key)

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, example in enumerate(examples, start=1):
            question_id = str(example["question_id"])
            user_id = index_example(store, embedding_client, example, args.granularity)
            retrieved = retriever.retrieve(
                user_id=user_id,
                query=str(example["question"]),
                top_k=args.top_k,
                min_score=args.min_score,
            )
            prompt = build_answer_prompt(example, retrieved)
            hypothesis = llm_client.generate(
                prompt=prompt,
                user_message=str(example["question"]),
                relevant_memories=retrieved,
            )
            output_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "hypothesis": hypothesis,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()
            print(f"[{index}/{len(examples)}] wrote hypothesis for {question_id}")

    print(f"Wrote hypotheses to {output_path}")


if __name__ == "__main__":
    main()
