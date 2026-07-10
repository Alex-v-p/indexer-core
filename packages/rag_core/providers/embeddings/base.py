from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Minimal async embedding provider contract."""

    vector_size: int

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
