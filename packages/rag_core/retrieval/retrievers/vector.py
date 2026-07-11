from __future__ import annotations

import uuid
from typing import Any, Protocol

from packages.rag_core.providers.embeddings import EmbeddingProvider
from packages.rag_core.providers.vector_stores import VectorSearchResult
from packages.rag_core.retrieval.models import EvidenceItem


class SearchableVectorStore(Protocol):
    """Vector-store capability needed by the baseline retriever."""

    async def ensure_collection(self) -> None:
        """Ensure the searchable collection exists before querying."""

    async def search_by_vector(self, vector: list[float], *, top_k: int) -> list[VectorSearchResult]:
        """Return ranked nearest-neighbour matches."""


class VectorRetriever:
    """Baseline dense-vector retriever.

    The retriever owns query embedding + vector-store search and returns
    normalized EvidenceItem objects for graph nodes. It deliberately does not
    depend on FastAPI, SQLAlchemy, or Qdrant-specific response models, which
    keeps it reusable for future agentic pipelines and evaluation harnesses.
    """

    def __init__(self, *, embedding_provider: EmbeddingProvider, vector_store: SearchableVectorStore) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embeddings = await self._embedding_provider.embed_texts([question])
        if not query_embeddings:
            return []

        await self._vector_store.ensure_collection()
        hits = await self._vector_store.search_by_vector(query_embeddings[0], top_k=top_k)
        return [_to_evidence_item(rank=rank, hit=hit) for rank, hit in enumerate(hits, start=1)]


def _to_evidence_item(*, rank: int, hit: VectorSearchResult) -> EvidenceItem:
    payload = dict(hit.payload)
    text = _payload_text(payload)
    metadata = _evidence_metadata(payload=payload, hit=hit)

    return EvidenceItem(
        rank=rank,
        text=text,
        score=hit.score,
        qdrant_chunk_index_id=_optional_uuid(payload.get("qdrant_chunk_index_id")),
        document_id=_optional_uuid(payload.get("document_id")),
        document_version_id=_optional_uuid(payload.get("document_version_id")),
        metadata=metadata,
    )


def _payload_text(payload: dict[str, Any]) -> str:
    value = payload.get("text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _evidence_metadata(*, payload: dict[str, Any], hit: VectorSearchResult) -> dict[str, Any]:
    metadata = {key: value for key, value in payload.items() if key != "text"}
    metadata["qdrant_point_id"] = hit.id
    metadata["retrieval_source"] = "vector"
    if hit.score is not None:
        metadata["score"] = hit.score
    return metadata


def _optional_uuid(value: object) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
