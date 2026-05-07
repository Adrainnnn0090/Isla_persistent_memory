from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from isla_memory.config import MemoryConfig
from isla_memory.embedding_client import HashEmbeddingClient
from isla_memory.memory_store import MemoryStore
from isla_memory.models import CandidateMemory, Memory, Message
from scripts import run_longmemeval_memory as lme


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[Message, Message | None]] = []

    def extract_memories(
        self,
        user_id: str,
        recent_messages: list[Message],
        current_user_message: Message,
        current_assistant_message: Message | None = None,
    ) -> list[CandidateMemory]:
        del user_id, recent_messages
        self.calls.append((current_user_message, current_assistant_message))
        source_message_id = (
            current_assistant_message.message_id
            if current_assistant_message is not None
            else current_user_message.message_id
        )
        return [
            CandidateMemory(
                content="The user's project deadline is Friday.",
                memory_type="fact",
                confidence=0.95,
                source_message_id=source_message_id,
                metadata={"source_role": "assistant"},
            )
        ]


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        user_message: str,
        relevant_memories: list[Memory],
    ) -> str:
        del user_message, relevant_memories
        self.prompts.append(prompt)
        return "Friday."


class _RecentLengthExtractor:
    def extract_memories(
        self,
        user_id: str,
        recent_messages: list[Message],
        current_user_message: Message,
        current_assistant_message: Message | None = None,
    ) -> list[CandidateMemory]:
        del user_id, current_assistant_message
        return [
            CandidateMemory(
                content=f"recent_messages={len(recent_messages)}",
                memory_type="fact",
                confidence=0.95,
                source_message_id=current_user_message.message_id,
            )
        ]


class LongMemEvalMemoryTest(unittest.TestCase):
    def test_memory_benchmark_replays_through_extractor_and_outputs_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "longmemeval.json"
            output_path = Path(tmp_dir) / "out.jsonl"
            db_path = Path(tmp_dir) / "memory.sqlite3"
            data_path.write_text(json.dumps([self._example()]), encoding="utf-8")
            extractor = _FakeExtractor()
            llm = _FakeLLM()

            lme.run_benchmark(
                args=self._args(data_path, output_path, db_path),
                config=MemoryConfig(),
                embedding_client=HashEmbeddingClient(),
                extractor=extractor,
                llm_client=llm,
            )

            output = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            memories = MemoryStore(str(db_path)).list_memories("q1")

            self.assertEqual(output, [{"question_id": "q1", "hypothesis": "Friday."}])
            self.assertEqual(len(extractor.calls), 1)
            self.assertIsNone(extractor.calls[0][1])
            self.assertIn("Assistant: The project deadline is Friday.", extractor.calls[0][0].content)
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].content, "The user's project deadline is Friday.")
            self.assertNotIn("When is the deadline?", memories[0].content)
            self.assertEqual(memories[0].metadata["source_date"], "2023/05/20 (Sat) 02:21")
            self.assertEqual(memories[0].metadata["extraction_granularity"], "chunk")
            self.assertIn("Retrieved Isla memories", llm.prompts[0])

    def test_chunk_granularity_batches_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "longmemeval.json"
            output_path = Path(tmp_dir) / "out.jsonl"
            db_path = Path(tmp_dir) / "memory.sqlite3"
            example = self._example()
            session_turns = []
            for index in range(5):
                session_turns.extend(
                    [
                        {"role": "user", "content": f"Question {index}?"},
                        {"role": "assistant", "content": f"Answer {index}."},
                    ]
                )
            example["haystack_sessions"] = [session_turns]
            data_path.write_text(json.dumps([example]), encoding="utf-8")
            extractor = _FakeExtractor()
            args = self._args(data_path, output_path, db_path)
            args.chunk_pairs = 2
            args.ingest_mode = "add-only"

            lme.run_benchmark(
                args=args,
                config=MemoryConfig(),
                embedding_client=HashEmbeddingClient(),
                extractor=extractor,
                llm_client=_FakeLLM(),
            )

            self.assertEqual(len(extractor.calls), 3)
            self.assertIn("Pair 0:", extractor.calls[0][0].content)
            self.assertIn("Pair 1:", extractor.calls[0][0].content)
            self.assertIn("Pair 4:", extractor.calls[2][0].content)

    def test_add_only_concurrent_extraction_preserves_chunk_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "longmemeval.json"
            output_path = Path(tmp_dir) / "out.jsonl"
            db_path = Path(tmp_dir) / "memory.sqlite3"
            example = self._example()
            session_turns = []
            for index in range(5):
                session_turns.extend(
                    [
                        {"role": "user", "content": f"Question {index}?"},
                        {"role": "assistant", "content": f"Answer {index}."},
                    ]
                )
            example["haystack_sessions"] = [session_turns]
            data_path.write_text(json.dumps([example]), encoding="utf-8")
            extractor = _FakeExtractor()
            args = self._args(data_path, output_path, db_path)
            args.chunk_pairs = 2
            args.ingest_mode = "add-only"
            args.extract_concurrency = 3

            lme.run_benchmark(
                args=args,
                config=MemoryConfig(),
                embedding_client=HashEmbeddingClient(),
                extractor=extractor,
                llm_client=_FakeLLM(),
            )

            memories = MemoryStore(str(db_path)).list_memories("q1")
            batch_starts = [memory.metadata["batch_start_pair"] for memory in memories]

            self.assertEqual(len(extractor.calls), 3)
            self.assertEqual(sorted(batch_starts), [0, 2, 4])

    def test_add_only_concurrent_extraction_uses_question_level_job_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "longmemeval.json"
            output_path = Path(tmp_dir) / "out.jsonl"
            db_path = Path(tmp_dir) / "memory.sqlite3"
            example = self._example()
            example["haystack_session_ids"] = ["s1", "s2", "s3"]
            example["haystack_dates"] = [
                "2023/05/20 (Sat) 02:21",
                "2023/05/21 (Sun) 02:21",
                "2023/05/22 (Mon) 02:21",
            ]
            example["haystack_sessions"] = [
                [
                    {"role": "user", "content": "Question 1?"},
                    {"role": "assistant", "content": "Answer 1."},
                ],
                [
                    {"role": "user", "content": "Question 2?"},
                    {"role": "assistant", "content": "Answer 2."},
                ],
                [
                    {"role": "user", "content": "Question 3?"},
                    {"role": "assistant", "content": "Answer 3."},
                ],
            ]
            data_path.write_text(json.dumps([example]), encoding="utf-8")
            args = self._args(data_path, output_path, db_path)
            args.ingest_mode = "add-only"
            args.extract_concurrency = 3
            observed_job_counts: list[int] = []
            original_runner = lme._run_extraction_jobs_concurrently

            def recording_runner(**kwargs):
                observed_job_counts.append(len(kwargs["jobs"]))
                return original_runner(**kwargs)

            with mock.patch.object(
                lme,
                "_run_extraction_jobs_concurrently",
                side_effect=recording_runner,
            ):
                lme.run_benchmark(
                    args=args,
                    config=MemoryConfig(),
                    embedding_client=HashEmbeddingClient(),
                    extractor=_RecentLengthExtractor(),
                    llm_client=_FakeLLM(),
                )

            memories = MemoryStore(str(db_path)).list_memories("q1")

            self.assertEqual(observed_job_counts, [3])
            self.assertEqual([memory.metadata["session_id"] for memory in memories], ["s1", "s2", "s3"])
            self.assertEqual(
                [memory.content for memory in memories],
                [
                    "recent_messages=0",
                    "recent_messages=2",
                    "recent_messages=4",
                ],
            )

    def test_resume_skips_completed_question_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "longmemeval.json"
            output_path = Path(tmp_dir) / "out.jsonl"
            db_path = Path(tmp_dir) / "memory.sqlite3"
            data_path.write_text(json.dumps([self._example()]), encoding="utf-8")
            output_path.write_text('{"question_id": "q1", "hypothesis": "existing"}\n', encoding="utf-8")
            extractor = _FakeExtractor()

            args = self._args(data_path, output_path, db_path)
            args.resume = True
            lme.run_benchmark(
                args=args,
                config=MemoryConfig(),
                embedding_client=HashEmbeddingClient(),
                extractor=extractor,
                llm_client=_FakeLLM(),
            )

            self.assertEqual(extractor.calls, [])
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{"question_id": "q1", "hypothesis": "existing"}\n',
            )

    @staticmethod
    def _args(data_path: Path, output_path: Path, db_path: Path) -> argparse.Namespace:
        return argparse.Namespace(
            data=str(data_path),
            output=str(output_path),
            limit=1,
            top_k=20,
            min_score=0.0,
            question_type=None,
            db_path=str(db_path),
            resume=False,
            ingest_mode="updater",
            include_assistant_facts=True,
            extraction_granularity="chunk",
            chunk_pairs=8,
            extract_concurrency=1,
        )

    @staticmethod
    def _example() -> dict[str, object]:
        return {
            "question_id": "q1",
            "question": "What is the user's project deadline?",
            "question_type": "single-session-assistant",
            "question_date": "2023/05/21 (Sun) 02:21",
            "answer": "Friday.",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "When is the deadline?"},
                    {"role": "assistant", "content": "The project deadline is Friday."},
                ]
            ],
        }


if __name__ == "__main__":
    unittest.main()
