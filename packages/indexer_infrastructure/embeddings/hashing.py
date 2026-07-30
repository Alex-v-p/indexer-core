from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)


class HashingEmbeddingProvider:
    """Deterministic local embedding provider for tests and offline development.

    This keeps ingestion runnable without an external embedding model. It is not
    meant as the final retrieval-quality implementation; the same interface can
    later be backed by Ollama, OpenAI, sentence-transformers, or another local
    model without changing ingestion or graph code.
    """

    def __init__(self, *, vector_size: int = 384) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")
        self.vector_size = vector_size

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embedding(text, self.vector_size) for text in texts]


def _hash_embedding(text: str, vector_size: int) -> list[float]:
    vector = [0.0] * vector_size
    tokens = _TOKEN_RE.findall(text.lower()) or [text.lower()]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % vector_size
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
