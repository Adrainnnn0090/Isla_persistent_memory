from __future__ import annotations

from collections.abc import Sequence

from isla_memory.config import MemoryConfig
from isla_memory.embedding_client import EmbeddingClient, HashEmbeddingClient
from isla_memory.llm_client import LLMClient, RuleBasedLLMClient
from isla_memory.memory_extractor import RuleBasedMemoryExtractor
from isla_memory.memory_retriever import MemoryRetriever
from isla_memory.memory_store import MemoryStore
from isla_memory.memory_updater import MemoryUpdater
from isla_memory.models import Memory, MemoryDecision, Message
from isla_memory.prompts import build_augmented_prompt
from isla_memory.utils import stable_id, utc_now


class MemoryAgent:
    def __init__(
        self,
        user_id: str,
        config: MemoryConfig | None = None,
        store: MemoryStore | None = None,
        embedding_client: EmbeddingClient | None = None,
        llm_client: LLMClient | None = None,
        recent_messages: Sequence[Message] | None = None,
    ) -> None:
        self.user_id = user_id
        self.config = config or MemoryConfig.from_env()
        self.embedding_client = embedding_client or HashEmbeddingClient(
            dimension=self.config.hash_embedding_dimension
        )
        self.store = store or MemoryStore(self.config.db_path)
        self.extractor = RuleBasedMemoryExtractor()
        self.updater = MemoryUpdater(
            store=self.store,
            embedding_client=self.embedding_client,
            min_confidence=self.config.min_confidence,
            dedup_score=self.config.dedup_score,
            update_score=self.config.update_score,
        )
        self.retriever = MemoryRetriever(
            store=self.store,
            embedding_client=self.embedding_client,
            min_score=self.config.min_score,
        )
        self.llm_client = llm_client or RuleBasedLLMClient()
        self.recent_messages = list(recent_messages or [])
        self.last_prompt = ""
        self.last_decisions: list[MemoryDecision] = []
        self.last_retrieved_memories: list[Memory] = []

    def chat(self, user_message: str) -> str:
        current_user_message = Message(
            message_id=stable_id("msg"),
            user_id=self.user_id,
            role="user",
            content=user_message,
            created_at=utc_now(),
        )

        relevant_memories = self.retriever.retrieve(
            user_id=self.user_id,
            query=user_message,
            top_k=self.config.top_k,
        )
        self.last_retrieved_memories = relevant_memories
        self.last_prompt = build_augmented_prompt(user_message, relevant_memories)

        response = self.llm_client.generate(
            prompt=self.last_prompt,
            user_message=user_message,
            relevant_memories=relevant_memories,
        )
        current_assistant_message = Message(
            message_id=stable_id("msg"),
            user_id=self.user_id,
            role="assistant",
            content=response,
            created_at=utc_now(),
        )

        candidates = self.extractor.extract_memories(
            user_id=self.user_id,
            recent_messages=self.recent_messages,
            current_user_message=current_user_message,
            current_assistant_message=current_assistant_message,
        )
        self.last_decisions = self.updater.update_memories(self.user_id, candidates)

        self.recent_messages.extend([current_user_message, current_assistant_message])
        self.recent_messages = self.recent_messages[-20:]
        return response

    def list_memories(self, include_invalid: bool = False) -> list[Memory]:
        return self.store.list_memories(self.user_id, include_invalid=include_invalid)
