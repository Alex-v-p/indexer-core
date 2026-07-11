from __future__ import annotations

from typing import Any

import httpx

from packages.rag_core.ports.embeddings import EmbeddingProviderError


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
