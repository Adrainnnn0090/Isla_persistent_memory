from __future__ import annotations

import hashlib
import re
from typing import Protocol

from isla_memory.utils import contains_any


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]:
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
