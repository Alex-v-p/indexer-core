from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

import httpx

_TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)


class EmbeddingProvider(Protocol):
    """Minimal async embedding provider contract."""

    vector_size: int

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot produce valid vectors."""


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


class OllamaEmbeddingProvider:
    """Embedding provider backed by Ollama's `/api/embed` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        vector_size: int,
        timeout_seconds: float = 120.0,
        truncate: bool = True,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")
        if not model.strip():
            raise ValueError("model must not be empty.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vector_size = vector_size
        self.timeout_seconds = timeout_seconds
        self.truncate = truncate

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "truncate": self.truncate,
            # Many embedding models support explicit dimensions. Keeping this
            # configured here guarantees Qdrant collection sizing remains stable
            # when the selected model supports it.
            "dimensions": self.vector_size,
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/embed", json=payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingProviderError(f"Ollama embedding request failed: {exc.response.text}") from exc

        body = response.json()
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingProviderError("Ollama embedding response did not include an embeddings array.")
        if len(embeddings) != len(texts):
            raise EmbeddingProviderError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} input texts.",
            )

        normalized_embeddings: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list):
                raise EmbeddingProviderError("Ollama returned a malformed embedding vector.")
            vector = [float(value) for value in embedding]
            if len(vector) != self.vector_size:
                raise EmbeddingProviderError(
                    "Ollama returned embedding vectors with size "
                    f"{len(vector)}, but EMBEDDING_VECTOR_SIZE is {self.vector_size}.",
                )
            normalized_embeddings.append(vector)

        return normalized_embeddings


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
