from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isla_memory.config import MemoryConfig
from isla_memory.embedding_client import (
    BGEEmbeddingClient,
    EmbeddingClient,
    HashEmbeddingClient,
    OpenAIEmbeddingClient,
)
from isla_memory.llm_client import LLMClient, OpenAILLMClient
from isla_memory.memory_extractor import OpenAIMemoryExtractor, RuleBasedMemoryExtractor
from isla_memory.memory_retriever import MemoryRetriever
from isla_memory.memory_store import MemoryStore
from isla_memory.memory_updater import MemoryUpdater
from isla_memory.models import CandidateMemory, Memory, Message
from isla_memory.utils import parse_datetime, stable_id, utc_now


@dataclass(slots=True)
class ExtractionJob:
    index: int
    session_id: str
    source_date: str
    pairs: list[tuple[Message, Message | None]]
    batch_start_pair: int
    extraction_granularity: str
    recent_messages: list[Message]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LongMemEval hypotheses by replaying conversations through Isla memory.",
    )
    parser.add_argument("--data", required=True, help="Path to a LongMemEval json file.")
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of retrieved memories.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Retrieval threshold.")
    parser.add_argument(
        "--question-type",
        action="append",
        default=None,
        help="Filter to one or more question_type values. Can be repeated.",
    )
    parser.add_argument(
        "--db-path",
        default="./data/longmemeval_memory_eval.sqlite3",
        help="Temporary SQLite path for the benchmark run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip question_ids already present in the output jsonl and keep the existing DB.",
    )
    parser.add_argument(
        "--ingest-mode",
        choices=("updater", "add-only"),
        default="updater",
        help="Use Isla updater or Mem0-style ADD-only ingestion.",
    )
    parser.add_argument(
        "--include-assistant-facts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow benchmark extraction from assistant messages.",
    )
    parser.add_argument(
        "--extraction-granularity",
        choices=("pair", "chunk", "session"),
        default="chunk",
        help="Extractor call granularity. Default chunks 8 user/assistant pairs per call.",
    )
    parser.add_argument(
        "--chunk-pairs",
        type=int,
        default=8,
        help="Number of user/assistant pairs per extractor call when using chunk granularity.",
    )
    parser.add_argument(
        "--extract-concurrency",
        type=int,
        default=1,
        help="Concurrent extractor calls for add-only ingestion. 1 keeps sequential execution.",
    )
    return parser.parse_args()


def build_embedding_client(config: MemoryConfig) -> EmbeddingClient:
    if config.embedding_provider == "openai":
        return OpenAIEmbeddingClient(
            model=config.embedding_model,
            api_key=config.openai_api_key,
        )
    if config.embedding_provider == "bge_m3":
        return BGEEmbeddingClient(
            model=config.bge_model,
            device=config.bge_device,
            batch_size=config.bge_batch_size,
            max_length=config.bge_max_length,
            use_fp16=config.bge_use_fp16,
        )
    return HashEmbeddingClient(dimension=config.hash_embedding_dimension)


def build_extractor(
    config: MemoryConfig,
    include_assistant_facts: bool,
) -> RuleBasedMemoryExtractor | OpenAIMemoryExtractor:
    if config.extractor_provider == "openai":
        return OpenAIMemoryExtractor(
            model=config.extractor_model,
            api_key=config.openai_api_key,
            include_assistant_facts=include_assistant_facts,
        )
    if config.extractor_provider != "rules":
        raise ValueError(f"Unsupported extractor provider: {config.extractor_provider}")
    return RuleBasedMemoryExtractor()


def load_examples(path: str, limit: int | None, question_types: list[str] | None) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        examples = json.load(file)
    if question_types:
        allowed = set(question_types)
        examples = [item for item in examples if item.get("question_type") in allowed]
    if limit is not None:
        examples = examples[:limit]
    return examples


def load_completed_question_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            question_id = item.get("question_id")
            if question_id:
                completed.add(str(question_id))
    return completed


def replay_example(
    *,
    store: MemoryStore,
    embedding_client: EmbeddingClient,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor,
    example: dict[str, Any],
    ingest_mode: str,
    extraction_granularity: str,
    chunk_pairs: int,
    extract_concurrency: int,
    config: MemoryConfig,
) -> str:
    user_id = str(example["question_id"])
    updater = MemoryUpdater(
        store=store,
        embedding_client=embedding_client,
        min_confidence=config.min_confidence,
        dedup_score=config.dedup_score,
        update_score=config.update_score,
        update_strategy=config.update_strategy,
        update_top_s=config.update_top_s,
        decision_min_confidence=config.decision_min_confidence,
        decision_fallback=config.decision_fallback,
    )
    recent_messages: list[Message] = []

    if ingest_mode == "add-only" and extract_concurrency > 1:
        _replay_example_add_only_concurrent(
            store=store,
            embedding_client=embedding_client,
            extractor=extractor,
            user_id=user_id,
            question_id=str(example["question_id"]),
            example=example,
            extraction_granularity=extraction_granularity,
            chunk_pairs=chunk_pairs,
            min_confidence=config.min_confidence,
            extract_concurrency=extract_concurrency,
        )
        return user_id

    for session_index, session in enumerate(example.get("haystack_sessions", [])):
        session_ids = example.get("haystack_session_ids", [])
        dates = example.get("haystack_dates", [])
        session_id = str(session_ids[session_index]) if session_index < len(session_ids) else str(session_index)
        source_date = str(dates[session_index]) if session_index < len(dates) else ""
        replay_session(
            store=store,
            embedding_client=embedding_client,
            extractor=extractor,
            updater=updater,
            user_id=user_id,
            question_id=str(example["question_id"]),
            session=session,
            session_id=session_id,
            source_date=source_date,
            ingest_mode=ingest_mode,
            extraction_granularity=extraction_granularity,
            chunk_pairs=chunk_pairs,
            extract_concurrency=extract_concurrency,
            recent_messages=recent_messages,
            min_confidence=config.min_confidence,
        )
        recent_messages[:] = recent_messages[-20:]

    return user_id


def _replay_example_add_only_concurrent(
    *,
    store: MemoryStore,
    embedding_client: EmbeddingClient,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor,
    user_id: str,
    question_id: str,
    example: dict[str, Any],
    extraction_granularity: str,
    chunk_pairs: int,
    min_confidence: float,
    extract_concurrency: int,
) -> None:
    jobs: list[ExtractionJob] = []
    recent_messages: list[Message] = []
    session_ids = example.get("haystack_session_ids", [])
    dates = example.get("haystack_dates", [])

    for session_index, session in enumerate(example.get("haystack_sessions", [])):
        session_id = str(session_ids[session_index]) if session_index < len(session_ids) else str(session_index)
        source_date = str(dates[session_index]) if session_index < len(dates) else ""
        pairs = _collect_session_pairs(
            user_id=user_id,
            session_id=session_id,
            session=session,
            source_date=source_date,
        )
        jobs.extend(
            _build_extraction_jobs(
                pairs=pairs,
                session_id=session_id,
                source_date=source_date,
                extraction_granularity=extraction_granularity,
                chunk_pairs=chunk_pairs,
                recent_messages=recent_messages,
                start_job_index=len(jobs),
            )
        )
        for user_message, assistant_message in pairs:
            _append_pair_messages(recent_messages, user_message, assistant_message)
        recent_messages[:] = recent_messages[-20:]

    results = _run_extraction_jobs_concurrently(
        extractor=extractor,
        user_id=user_id,
        question_id=question_id,
        jobs=jobs,
        extract_concurrency=extract_concurrency,
    )
    _apply_add_only_results(
        store=store,
        embedding_client=embedding_client,
        user_id=user_id,
        jobs=jobs,
        results=results,
        min_confidence=min_confidence,
    )


def replay_session(
    *,
    store: MemoryStore,
    embedding_client: EmbeddingClient,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor,
    updater: MemoryUpdater,
    user_id: str,
    question_id: str,
    session: list[dict[str, Any]],
    session_id: str,
    source_date: str,
    ingest_mode: str,
    extraction_granularity: str,
    chunk_pairs: int,
    extract_concurrency: int,
    recent_messages: list[Message],
    min_confidence: float,
) -> None:
    pairs = _collect_session_pairs(
        user_id=user_id,
        session_id=session_id,
        session=session,
        source_date=source_date,
    )
    if ingest_mode == "add-only" and extract_concurrency > 1:
        _replay_session_add_only_concurrent(
            store=store,
            embedding_client=embedding_client,
            extractor=extractor,
            user_id=user_id,
            question_id=question_id,
            session_id=session_id,
            source_date=source_date,
            pairs=pairs,
            extraction_granularity=extraction_granularity,
            chunk_pairs=chunk_pairs,
            recent_messages=recent_messages,
            min_confidence=min_confidence,
            extract_concurrency=extract_concurrency,
        )
        return

    if extraction_granularity == "pair":
        for user_message, assistant_message in pairs:
            _ingest_pair(
                store=store,
                embedding_client=embedding_client,
                extractor=extractor,
                updater=updater,
                user_id=user_id,
                question_id=question_id,
                session_id=session_id,
                source_date=source_date,
                user_message=user_message,
                assistant_message=assistant_message,
                recent_messages=recent_messages,
                ingest_mode=ingest_mode,
                min_confidence=min_confidence,
            )
            _append_pair_messages(recent_messages, user_message, assistant_message)
        return

    size = len(pairs) if extraction_granularity == "session" else max(chunk_pairs, 1)
    for start_index in range(0, len(pairs), size):
        batch = pairs[start_index : start_index + size]
        _ingest_pair_batch(
            store=store,
            embedding_client=embedding_client,
            extractor=extractor,
            updater=updater,
            user_id=user_id,
            question_id=question_id,
            session_id=session_id,
            source_date=source_date,
            pairs=batch,
            batch_start_pair=start_index,
            extraction_granularity=extraction_granularity,
            recent_messages=recent_messages,
            ingest_mode=ingest_mode,
            min_confidence=min_confidence,
        )
        for user_message, assistant_message in batch:
            _append_pair_messages(recent_messages, user_message, assistant_message)


def _replay_session_add_only_concurrent(
    *,
    store: MemoryStore,
    embedding_client: EmbeddingClient,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor,
    user_id: str,
    question_id: str,
    session_id: str,
    source_date: str,
    pairs: list[tuple[Message, Message | None]],
    extraction_granularity: str,
    chunk_pairs: int,
    recent_messages: list[Message],
    min_confidence: float,
    extract_concurrency: int,
) -> None:
    jobs = _build_extraction_jobs(
        pairs=pairs,
        session_id=session_id,
        source_date=source_date,
        extraction_granularity=extraction_granularity,
        chunk_pairs=chunk_pairs,
        recent_messages=recent_messages,
        start_job_index=0,
    )
    if not jobs:
        return

    results = _run_extraction_jobs_concurrently(
        extractor=extractor,
        user_id=user_id,
        question_id=question_id,
        jobs=jobs,
        extract_concurrency=extract_concurrency,
    )
    _apply_add_only_results(
        store=store,
        embedding_client=embedding_client,
        user_id=user_id,
        jobs=jobs,
        results=results,
        min_confidence=min_confidence,
    )
    for job in jobs:
        for user_message, assistant_message in job.pairs:
            _append_pair_messages(recent_messages, user_message, assistant_message)


def _build_extraction_jobs(
    *,
    pairs: list[tuple[Message, Message | None]],
    session_id: str,
    source_date: str,
    extraction_granularity: str,
    chunk_pairs: int,
    recent_messages: list[Message],
    start_job_index: int,
) -> list[ExtractionJob]:
    jobs: list[ExtractionJob] = []
    recent_snapshot = list(recent_messages)

    if extraction_granularity == "pair":
        for index, pair in enumerate(pairs):
            jobs.append(
                ExtractionJob(
                    index=start_job_index + index,
                    session_id=session_id,
                    source_date=source_date,
                    pairs=[pair],
                    batch_start_pair=index,
                    extraction_granularity=extraction_granularity,
                    recent_messages=list(recent_snapshot),
                )
            )
            _append_pair_messages(recent_snapshot, pair[0], pair[1])
        return jobs

    size = len(pairs) if extraction_granularity == "session" else max(chunk_pairs, 1)
    for index, start_index in enumerate(range(0, len(pairs), size)):
        batch = pairs[start_index : start_index + size]
        jobs.append(
            ExtractionJob(
                index=start_job_index + index,
                session_id=session_id,
                source_date=source_date,
                pairs=batch,
                batch_start_pair=start_index,
                extraction_granularity=extraction_granularity,
                recent_messages=list(recent_snapshot),
            )
        )
        for user_message, assistant_message in batch:
            _append_pair_messages(recent_snapshot, user_message, assistant_message)
    return jobs


def _run_extraction_jobs_concurrently(
    *,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor,
    user_id: str,
    question_id: str,
    jobs: list[ExtractionJob],
    extract_concurrency: int,
) -> dict[int, list[CandidateMemory]]:
    if not jobs:
        return {}

    print(f"question-level extraction jobs: {len(jobs)} with concurrency {extract_concurrency}")
    results: dict[int, list[CandidateMemory]] = {}
    with ThreadPoolExecutor(max_workers=max(extract_concurrency, 1)) as executor:
        futures = {
            executor.submit(
                _extract_job_candidates,
                extractor=extractor,
                user_id=user_id,
                question_id=question_id,
                job=job,
            ): job.index
            for job in jobs
        }
        for future in as_completed(futures):
            job_index, candidates = future.result()
            results[job_index] = candidates
    return results


def _apply_add_only_results(
    *,
    store: MemoryStore,
    embedding_client: EmbeddingClient,
    user_id: str,
    jobs: list[ExtractionJob],
    results: dict[int, list[CandidateMemory]],
    min_confidence: float,
) -> None:
    for job in jobs:
        for candidate in results.get(job.index, []):
            if candidate.confidence >= min_confidence:
                add_candidate_memory(store, embedding_client, user_id, candidate)


def _extract_job_candidates(
    *,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor,
    user_id: str,
    question_id: str,
    job: ExtractionJob,
) -> tuple[int, list[CandidateMemory]]:
    if job.extraction_granularity == "pair":
        user_message, assistant_message = job.pairs[0]
        candidates = extractor.extract_memories(
            user_id=user_id,
            recent_messages=job.recent_messages,
            current_user_message=user_message,
            current_assistant_message=assistant_message,
        )
        enriched = [
            _with_benchmark_metadata(
                candidate=candidate,
                question_id=question_id,
                session_id=job.session_id,
                source_date=job.source_date,
                source_role="user_assistant_pair" if assistant_message is not None else user_message.role,
            )
            for candidate in candidates
        ]
        return job.index, enriched

    batch_message = _batch_message_from_pairs(
        user_id=user_id,
        session_id=job.session_id,
        source_date=job.source_date,
        pairs=job.pairs,
        batch_start_pair=job.batch_start_pair,
        extraction_granularity=job.extraction_granularity,
    )
    candidates = extractor.extract_memories(
        user_id=user_id,
        recent_messages=job.recent_messages,
        current_user_message=batch_message,
        current_assistant_message=None,
    )
    enriched = [
            _with_benchmark_metadata(
                candidate=candidate,
                question_id=question_id,
                session_id=job.session_id,
                source_date=job.source_date,
                source_role=job.extraction_granularity,
                extraction_granularity=job.extraction_granularity,
            batch_start_pair=job.batch_start_pair,
            batch_pair_count=len(job.pairs),
        )
        for candidate in candidates
    ]
    return job.index, enriched


def _collect_session_pairs(
    *,
    user_id: str,
    session_id: str,
    session: list[dict[str, Any]],
    source_date: str,
) -> list[tuple[Message, Message | None]]:
    pairs: list[tuple[Message, Message | None]] = []
    turn_index = 0
    while turn_index < len(session):
        turn = session[turn_index]
        role = str(turn.get("role", "")).lower()
        if role == "user":
            user_message = _message_from_turn(
                user_id=user_id,
                session_id=session_id,
                turn_index=turn_index,
                turn=turn,
                source_date=source_date,
            )
            assistant_message = None
            next_index = turn_index + 1
            if next_index < len(session) and str(session[next_index].get("role", "")).lower() == "assistant":
                assistant_message = _message_from_turn(
                    user_id=user_id,
                    session_id=session_id,
                    turn_index=next_index,
                    turn=session[next_index],
                    source_date=source_date,
                )
                turn_index += 2
            else:
                turn_index += 1
            pairs.append((user_message, assistant_message))
            continue

        if role == "assistant":
            assistant_message = _message_from_turn(
                user_id=user_id,
                session_id=session_id,
                turn_index=turn_index,
                turn=turn,
                source_date=source_date,
            )
            empty_user_message = Message(
                message_id=f"lme_msg_{session_id}_{turn_index}_empty_user",
                user_id=user_id,
                role="user",
                content="",
                created_at=assistant_message.created_at,
            )
            pairs.append((empty_user_message, assistant_message))
        turn_index += 1
    return pairs


def _append_pair_messages(
    recent_messages: list[Message],
    user_message: Message,
    assistant_message: Message | None,
) -> None:
    if user_message.content:
        recent_messages.append(user_message)
    if assistant_message is not None:
        recent_messages.append(assistant_message)


def _ingest_pair_batch(
    *,
    store: MemoryStore,
    embedding_client: EmbeddingClient,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor,
    updater: MemoryUpdater,
    user_id: str,
    question_id: str,
    session_id: str,
    source_date: str,
    pairs: list[tuple[Message, Message | None]],
    batch_start_pair: int,
    extraction_granularity: str,
    recent_messages: list[Message],
    ingest_mode: str,
    min_confidence: float,
) -> None:
    if not pairs:
        return
    batch_message = _batch_message_from_pairs(
        user_id=user_id,
        session_id=session_id,
        source_date=source_date,
        pairs=pairs,
        batch_start_pair=batch_start_pair,
        extraction_granularity=extraction_granularity,
    )
    candidates = extractor.extract_memories(
        user_id=user_id,
        recent_messages=recent_messages,
        current_user_message=batch_message,
        current_assistant_message=None,
    )
    enriched = [
        _with_benchmark_metadata(
            candidate=candidate,
            question_id=question_id,
            session_id=session_id,
            source_date=source_date,
            source_role=extraction_granularity,
            extraction_granularity=extraction_granularity,
            batch_start_pair=batch_start_pair,
            batch_pair_count=len(pairs),
        )
        for candidate in candidates
    ]
    if ingest_mode == "add-only":
        for candidate in enriched:
            if candidate.confidence >= min_confidence:
                add_candidate_memory(store, embedding_client, user_id, candidate)
        return
    updater.update_memories(user_id, enriched, current_user_message=batch_message)


def _batch_message_from_pairs(
    *,
    user_id: str,
    session_id: str,
    source_date: str,
    pairs: list[tuple[Message, Message | None]],
    batch_start_pair: int,
    extraction_granularity: str,
) -> Message:
    return Message(
        message_id=f"lme_{extraction_granularity}_{session_id}_{batch_start_pair}_{batch_start_pair + len(pairs) - 1}",
        user_id=user_id,
        role="user",
        content=_format_pair_batch(
            session_id=session_id,
            source_date=source_date,
            pairs=pairs,
            batch_start_pair=batch_start_pair,
            extraction_granularity=extraction_granularity,
        ),
        created_at=pairs[0][0].created_at,
    )


def _format_pair_batch(
    *,
    session_id: str,
    source_date: str,
    pairs: list[tuple[Message, Message | None]],
    batch_start_pair: int,
    extraction_granularity: str,
) -> str:
    lines = [
        f"[LongMemEval {extraction_granularity} transcript]",
        f"[session_id: {session_id}]",
        f"[date: {source_date}]",
    ]
    for offset, (user_message, assistant_message) in enumerate(pairs):
        pair_number = batch_start_pair + offset
        lines.append(f"Pair {pair_number}:")
        if user_message.content:
            lines.append(f"User: {user_message.content}")
        if assistant_message is not None:
            lines.append(f"Assistant: {assistant_message.content}")
    return "\n".join(lines)


def _ingest_pair(
    *,
    store: MemoryStore,
    embedding_client: EmbeddingClient,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor,
    updater: MemoryUpdater,
    user_id: str,
    question_id: str,
    session_id: str,
    source_date: str,
    user_message: Message,
    assistant_message: Message | None,
    recent_messages: list[Message],
    ingest_mode: str,
    min_confidence: float,
) -> None:
    candidates = extractor.extract_memories(
        user_id=user_id,
        recent_messages=recent_messages,
        current_user_message=user_message,
        current_assistant_message=assistant_message,
    )
    enriched = [
        _with_benchmark_metadata(
            candidate=candidate,
            question_id=question_id,
            session_id=session_id,
            source_date=source_date,
            source_role="user_assistant_pair" if assistant_message is not None else user_message.role,
        )
        for candidate in candidates
    ]
    if ingest_mode == "add-only":
        for candidate in enriched:
            if candidate.confidence >= min_confidence:
                add_candidate_memory(store, embedding_client, user_id, candidate)
        return
    updater.update_memories(user_id, enriched, current_user_message=user_message)


def _message_from_turn(
    *,
    user_id: str,
    session_id: str,
    turn_index: int,
    turn: dict[str, Any],
    source_date: str,
) -> Message:
    return Message(
        message_id=f"lme_msg_{session_id}_{turn_index}",
        user_id=user_id,
        role=str(turn.get("role", "user")).lower(),  # type: ignore[arg-type]
        content=str(turn.get("content", "")).strip(),
        created_at=_source_datetime(source_date) or utc_now(),
    )


def _with_benchmark_metadata(
    *,
    candidate: CandidateMemory,
    question_id: str,
    session_id: str,
    source_date: str,
    source_role: str,
    extraction_granularity: str = "pair",
    batch_start_pair: int | None = None,
    batch_pair_count: int | None = None,
) -> CandidateMemory:
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "benchmark": "longmemeval",
            "question_id": question_id,
            "session_id": session_id,
            "source_date": source_date,
            "source_role": source_role,
            "extraction_granularity": extraction_granularity,
        }
    )
    if batch_start_pair is not None:
        metadata["batch_start_pair"] = batch_start_pair
    if batch_pair_count is not None:
        metadata["batch_pair_count"] = batch_pair_count
    return CandidateMemory(
        content=candidate.content,
        memory_type=candidate.memory_type,
        confidence=candidate.confidence,
        source_message_id=candidate.source_message_id,
        metadata=metadata,
    )


def add_candidate_memory(
    store: MemoryStore,
    embedding_client: EmbeddingClient,
    user_id: str,
    candidate: CandidateMemory,
) -> None:
    source_time = _source_datetime(str(candidate.metadata.get("source_date", ""))) or utc_now()
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "memory_type": candidate.memory_type,
            "confidence": candidate.confidence,
            "source_message_id": candidate.source_message_id,
        }
    )
    store.add_memory(
        Memory(
            memory_id=stable_id("lme_mem"),
            user_id=user_id,
            content=candidate.content,
            embedding=embedding_client.embed(candidate.content),
            memory_type=candidate.memory_type,
            created_at=source_time,
            updated_at=source_time,
            source_message_id=candidate.source_message_id,
            metadata=metadata,
        )
    )


def _source_datetime(value: str):
    if not value:
        return None
    try:
        return parse_datetime(value)
    except ValueError:
        pass
    from datetime import UTC, datetime

    for date_format in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, date_format).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def build_answer_prompt(example: dict[str, Any], retrieved: list[Memory]) -> str:
    memory_text = "\n\n".join(
        f"Memory {index + 1}:\n{memory.content}"
        for index, memory in enumerate(retrieved)
    )
    if not memory_text:
        memory_text = "No relevant memory retrieved."

    return f"""You answer LongMemEval questions using only the retrieved Isla memories.

This is a memory-system evaluation, not an oracle raw-history retrieval baseline.

Question date:
{example.get("question_date", "")}

Retrieved Isla memories:
{memory_text}

Question:
{example["question"]}

Answer concisely. If the retrieved memories do not contain enough evidence, say you do not know based on the provided memories.
"""


def run_benchmark(
    *,
    args: argparse.Namespace,
    config: MemoryConfig,
    embedding_client: EmbeddingClient | None = None,
    extractor: RuleBasedMemoryExtractor | OpenAIMemoryExtractor | None = None,
    llm_client: LLMClient | None = None,
) -> None:
    db_path = Path(args.db_path)
    if db_path.exists() and not args.resume:
        db_path.unlink()

    examples = load_examples(args.data, args.limit, args.question_type)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_question_ids(output_path) if args.resume else set()
    if completed:
        print(f"Resume mode: skipping {len(completed)} completed question_ids")

    store = MemoryStore(str(db_path))
    embedding_client = embedding_client or build_embedding_client(config)
    extractor = extractor or build_extractor(config, args.include_assistant_facts)
    llm_client = llm_client or OpenAILLMClient(model=config.llm_model, api_key=config.openai_api_key)
    retriever = MemoryRetriever(store, embedding_client, min_score=args.min_score)
    if args.extract_concurrency > 1 and args.ingest_mode != "add-only":
        print("--extract-concurrency only applies to --ingest-mode add-only; updater ingestion remains sequential.")

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for index, example in enumerate(examples, start=1):
            question_id = str(example["question_id"])
            if question_id in completed:
                print(f"[{index}/{len(examples)}] skipping completed {question_id}")
                continue
            sessions = len(example.get("haystack_sessions", []))
            turns = sum(len(session) for session in example.get("haystack_sessions", []))
            print(
                f"[{index}/{len(examples)}] replaying {question_id}: "
                f"{sessions} sessions, {turns} turns"
            )
            user_id = replay_example(
                store=store,
                embedding_client=embedding_client,
                extractor=extractor,
                example=example,
                ingest_mode=args.ingest_mode,
                extraction_granularity=args.extraction_granularity,
                chunk_pairs=args.chunk_pairs,
                extract_concurrency=args.extract_concurrency,
                config=config,
            )
            print(f"[{index}/{len(examples)}] retrieving and answering {question_id}")
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


def main() -> None:
    args = parse_args()
    config = MemoryConfig.from_env()
    if config.llm_provider != "openai":
        raise SystemExit("Set MEMORY_LLM_PROVIDER=openai before running LongMemEval memory generation.")
    run_benchmark(args=args, config=config)


if __name__ == "__main__":
    main()
