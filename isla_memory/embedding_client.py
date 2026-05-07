from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any
from typing import Protocol

from isla_memory.utils import contains_any


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbeddingClient:
    """Deterministic local embedding for demos and tests.

    It mixes a small set of semantic keyword buckets with hashed lexical
    features. This is not a production embedding model, but it keeps the MVP
    runnable without network access or external dependencies.
    """

    _SEMANTIC_FEATURES: tuple[tuple[int, tuple[str, ...]], ...] = (
        (0, ("回答", "解释", "回复", "答复", "answer", "response", "style", "风格", "方式")),
        (1, ("偏好", "喜欢", "希望", "以后", "请", "prefer", "preference", "like")),
        (2, ("中文", "汉语", "chinese")),
        (3, ("英文", "english")),
        (4, ("直接", "简洁", "短一点", "短些", "concise", "brief", "direct")),
        (5, ("技术", "代码", "编程", "系统", "项目", "project", "programming")),
        (6, ("mem0", "记忆系统", "长期记忆", "memory", "vector", "embedding", "向量")),
        (7, ("正在做", "我在做", "构建", "实现", "开发", "build", "working")),
        (8, ("名字", "我叫", "name")),
        (9, ("咖啡", "coffee")),
        (10, ("取消", "忘记", "不要再记住", "不再记住", "delete", "forget")),
    )

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 64:
            raise ValueError("dimension must be at least 64")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        normalized = text.lower()

        for index, keywords in self._SEMANTIC_FEATURES:
            if contains_any(normalized, keywords):
                vector[index] += 4.0

        for token in self._tokens(normalized):
            vector[self._hash_index(token)] += 0.35

        for bigram in self._cjk_bigrams(normalized):
            vector[self._hash_index(bigram)] += 0.45

        return vector

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def _hash_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        raw = int.from_bytes(digest, byteorder="big")
        return 32 + raw % (self.dimension - 32)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)

    @staticmethod
    def _cjk_bigrams(text: str) -> list[str]:
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        return [chars[index] + chars[index + 1] for index in range(len(chars) - 1)]



class OpenAIEmbeddingClient:

    def __init__(

        self,

        model: str = "text-embedding-3-small",

        api_key: str | None = None,

    ) -> None:

        self.model = model

        try:

            from openai import OpenAI

        except ImportError as exc:

            raise RuntimeError(

                'OpenAI provider requires the "openai" package. '

                'Install it with: pip install -e ".[openai]"'

            ) from exc

        self.client = OpenAI(

            api_key=api_key,

            timeout=120,

            max_retries=5,

        )

    def embed(self, text: str) -> list[float]:

        from openai import APITimeoutError, APIConnectionError, RateLimitError

        last_err = None

        for attempt in range(5):

            try:

                response = self.client.embeddings.create(

                    model=self.model,

                    input=text,

                )

                return list(response.data[0].embedding)

            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:

                last_err = exc

                time.sleep(2 ** attempt)

        raise last_err

    def embed_many(self, texts: list[str]) -> list[list[float]]:

        from openai import APITimeoutError, APIConnectionError, RateLimitError

        if not texts:

            return []

        last_err = None

        for attempt in range(5):

            try:

                response = self.client.embeddings.create(

                    model=self.model,

                    input=texts,

                )

                return [list(item.embedding) for item in response.data]

            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:

                last_err = exc

                time.sleep(2 ** attempt)

        raise last_err


# class OpenAIEmbeddingClient:
#     def __init__(
#         self,
#         model: str = "text-embedding-3-small",
#         api_key: str | None = None,
#     ) -> None:
#         self.model = model
#         try:
#             from openai import OpenAI
#         except ImportError as exc:
#             raise RuntimeError(
#                 'OpenAI provider requires the "openai" package. '
#                 'Install it with: pip install -e ".[openai]"'
#             ) from exc
#         self.client = OpenAI(api_key=api_key)

#     def embed(self, text: str) -> list[float]:
#         response = self.client.embeddings.create(
#             model=self.model,
#             input=text,
#         )
#         return list(response.data[0].embedding)

#     def embed_many(self, texts: list[str]) -> list[list[float]]:
#         if not texts:
#             return []
#         response = self.client.embeddings.create(
#             model=self.model,
#             input=texts,
#         )
#         return [list(item.embedding) for item in response.data]







class BGEEmbeddingClient:
    """Local BGE-M3 embedding client backed by FlagEmbedding."""

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 8192,
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                'BGE-M3 provider requires the "FlagEmbedding" package. '
                'Install it with: pip install -e ".[local-embedding]"'
            ) from exc
        self.model = BGEM3FlagModel(
            model,
            use_fp16=use_fp16,
            device=self.device,
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._extract_dense_vectors(encoded, expected_count=len(texts))

    @staticmethod
    def _resolve_device(device: str) -> str | None:
        if device != "auto":
            return device
        try:
            import torch
        except ImportError:
            return None
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _extract_dense_vectors(encoded: Any, expected_count: int) -> list[list[float]]:
        dense_vectors = encoded.get("dense_vecs") if isinstance(encoded, dict) else encoded
        if expected_count == 1 and not BGEEmbeddingClient._looks_like_matrix(dense_vectors):
            return [BGEEmbeddingClient._to_float_list(dense_vectors)]
        return [BGEEmbeddingClient._to_float_list(vector) for vector in dense_vectors]

    @staticmethod
    def _looks_like_matrix(value: Any) -> bool:
        if hasattr(value, "ndim"):
            return int(value.ndim) > 1
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return False
        if not value:
            return False
        return isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes))

    @staticmethod
    def _to_float_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]
