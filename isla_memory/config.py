from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_PATH = "./data/memory.sqlite3"
DEFAULT_LLM_PROVIDER = "rules"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_PROVIDER = "hash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EXTRACTOR_PROVIDER = "rules"
DEFAULT_EXTRACTOR_MODEL = "gpt-4.1-mini"
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.35
DEFAULT_DEDUP_SCORE = 0.90
DEFAULT_UPDATE_SCORE = 0.62
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_HASH_EMBEDDING_DIMENSION = 256


def _read_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get_value(file_values: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key, file_values.get(key, default))


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    db_path: str = DEFAULT_DB_PATH
    openai_api_key: str | None = None
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    extractor_provider: str = DEFAULT_EXTRACTOR_PROVIDER
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    dedup_score: float = DEFAULT_DEDUP_SCORE
    update_score: float = DEFAULT_UPDATE_SCORE
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    hash_embedding_dimension: int = DEFAULT_HASH_EMBEDDING_DIMENSION

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "MemoryConfig":
        file_values = _read_env_file(env_file)
        return cls(
            db_path=_get_value(file_values, "MEMORY_DB_PATH", DEFAULT_DB_PATH),
            openai_api_key=_get_value(file_values, "OPENAI_API_KEY", "") or None,
            llm_provider=_get_value(file_values, "MEMORY_LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
            llm_model=_get_value(file_values, "MEMORY_LLM_MODEL", DEFAULT_LLM_MODEL),
            embedding_provider=_get_value(
                file_values,
                "MEMORY_EMBEDDING_PROVIDER",
                DEFAULT_EMBEDDING_PROVIDER,
            ),
            embedding_model=_get_value(
                file_values,
                "MEMORY_EMBEDDING_MODEL",
                DEFAULT_EMBEDDING_MODEL,
            ),
            extractor_provider=_get_value(
                file_values,
                "MEMORY_EXTRACTOR_PROVIDER",
                DEFAULT_EXTRACTOR_PROVIDER,
            ),
            extractor_model=_get_value(
                file_values,
                "MEMORY_EXTRACTOR_MODEL",
                DEFAULT_EXTRACTOR_MODEL,
            ),
            top_k=int(_get_value(file_values, "MEMORY_TOP_K", str(DEFAULT_TOP_K))),
            min_score=float(_get_value(file_values, "MEMORY_MIN_SCORE", str(DEFAULT_MIN_SCORE))),
            dedup_score=float(_get_value(file_values, "MEMORY_DEDUP_SCORE", str(DEFAULT_DEDUP_SCORE))),
            update_score=float(_get_value(file_values, "MEMORY_UPDATE_SCORE", str(DEFAULT_UPDATE_SCORE))),
            min_confidence=float(
                _get_value(file_values, "MEMORY_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE))
            ),
            hash_embedding_dimension=int(
                _get_value(
                    file_values,
                    "HASH_EMBEDDING_DIMENSION",
                    str(DEFAULT_HASH_EMBEDDING_DIMENSION),
                )
            ),
        )
