from __future__ import annotations

from app.core.config import Settings
from packages.rag_core.providers import QdrantVectorStore


def build_vector_store(settings: Settings) -> QdrantVectorStore:
    return QdrantVectorStore(
        base_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_vector_size,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
