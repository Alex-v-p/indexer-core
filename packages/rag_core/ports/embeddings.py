from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Async embedding capability required by ingestion and retrieval."""

    vector_size: int

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding implementation cannot produce valid vectors."""
