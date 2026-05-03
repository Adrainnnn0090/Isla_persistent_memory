from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from isla_memory.utils import datetime_to_iso, parse_datetime, utc_now

Role = Literal["user", "assistant", "system"]
MemoryType = Literal["preference", "fact", "goal", "constraint", "profile", "other"]
MemoryAction = Literal["ADD", "UPDATE", "DELETE", "NOOP"]


@dataclass(slots=True)
class Message:
    message_id: str
    user_id: str
    role: Role
    content: str
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "role": self.role,
            "content": self.content,
            "created_at": datetime_to_iso(self.created_at),
        }


@dataclass(slots=True)
class CandidateMemory:
    content: str
    memory_type: MemoryType
    confidence: float
    source_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_delete_intent(self) -> bool:
        return self.metadata.get("intent") == "delete"


@dataclass(slots=True)
class Memory:
    memory_id: str
    user_id: str
    content: str
    embedding: list[float]
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    source_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    invalid_at: datetime | None = None

    @property
    def is_valid(self) -> bool:
        return self.invalid_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "embedding": self.embedding,
            "created_at": datetime_to_iso(self.created_at),
            "updated_at": datetime_to_iso(self.updated_at),
            "source_message_id": self.source_message_id,
            "metadata": self.metadata,
            "invalid_at": datetime_to_iso(self.invalid_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        return cls(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            content=data["content"],
            embedding=list(data["embedding"]),
            created_at=parse_datetime(data.get("created_at")) or utc_now(),
            updated_at=parse_datetime(data.get("updated_at")) or utc_now(),
            source_message_id=data.get("source_message_id"),
            metadata=dict(data.get("metadata") or {}),
            invalid_at=parse_datetime(data.get("invalid_at")),
        )


@dataclass(slots=True)
class MemoryDecision:
    action: MemoryAction
    candidate: CandidateMemory
    target_memory_id: str | None = None
    final_content: str | None = None
    reason: str = ""
