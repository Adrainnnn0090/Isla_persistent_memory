from __future__ import annotations

from collections.abc import Sequence

from isla_memory.config import MemoryConfig
from isla_memory.embedding_client import EmbeddingClient, HashEmbeddingClient, OpenAIEmbeddingClient
from isla_memory.llm_client import LLMClient, OpenAILLMClient, RuleBasedLLMClient
from isla_memory.memory_extractor import OpenAIMemoryExtractor, RuleBasedMemoryExtractor
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
        self.embedding_client = embedding_client or self._build_embedding_client(self.config)
        self.store = store or MemoryStore(self.config.db_path)
        self.extractor = self._build_extractor(self.config)
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
        self.llm_client = llm_client or self._build_llm_client(self.config)
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

    @staticmethod
    def _build_embedding_client(config: MemoryConfig) -> EmbeddingClient:
        if config.embedding_provider == "openai":
            return OpenAIEmbeddingClient(
                model=config.embedding_model,
                api_key=config.openai_api_key,
            )
        if config.embedding_provider != "hash":
            raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
        return HashEmbeddingClient(dimension=config.hash_embedding_dimension)

    @staticmethod
    def _build_llm_client(config: MemoryConfig) -> LLMClient:
        if config.llm_provider == "openai":
            return OpenAILLMClient(
                model=config.llm_model,
                api_key=config.openai_api_key,
            )
        if config.llm_provider != "rules":
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
        return RuleBasedLLMClient()

    @staticmethod
    def _build_extractor(config: MemoryConfig) -> RuleBasedMemoryExtractor | OpenAIMemoryExtractor:
        if config.extractor_provider == "openai":
            return OpenAIMemoryExtractor(
                model=config.extractor_model,
                api_key=config.openai_api_key,
            )
        if config.extractor_provider != "rules":
            raise ValueError(f"Unsupported extractor provider: {config.extractor_provider}")
        return RuleBasedMemoryExtractor()
