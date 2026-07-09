from __future__ import annotations

from app.core.config import Settings
from packages.rag_core.providers import EmbeddingProvider, HashingEmbeddingProvider, OllamaEmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(vector_size=settings.embedding_vector_size)

    return OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        vector_size=settings.embedding_vector_size,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
