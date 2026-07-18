from __future__ import annotations

import uuid
from typing import Any, Protocol

from packages.rag_core.ports import EmbeddingProvider
from packages.rag_core.ports import VectorSearchResult
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
from packages.rag_core.retrieval.retrievers.base import callable_accepts_parameter


class SearchableVectorStore(Protocol):
    """Vector-store capability needed by dense retrievers."""

    async def ensure_collection(self) -> None:
        """Ensure the searchable collection exists before querying."""

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        vector_name: str,
        top_k: int,
        version_constraint=None,
        date_constraints=(),
    ) -> list[VectorSearchResult]:
        """Return ranked matches from one named vector representation."""


class VectorRetriever:
    """Dense-vector retriever over a configured named representation."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: SearchableVectorStore,
        vector_name: str,
    ) -> None:
        if not vector_name.strip():
            raise ValueError("vector_name must not be empty.")
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._vector_name = vector_name

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> list[EvidenceItem]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embeddings = await self._embedding_provider.embed_texts([question])
        if not query_embeddings:
            return []

        await self._vector_store.ensure_collection()
        search = self._vector_store.search_by_vector
        search_kwargs = {"vector_name": self._vector_name, "top_k": top_k}
        if constraints is not None and callable_accepts_parameter(search, "version_constraint"):
            search_kwargs["version_constraint"] = constraints.version
        if constraints is not None and callable_accepts_parameter(search, "date_constraints"):
            search_kwargs["date_constraints"] = constraints.dates
        hits = await search(query_embeddings[0], **search_kwargs)
        return [
            _to_evidence_item(rank=rank, hit=hit, vector_name=self._vector_name)
            for rank, hit in enumerate(hits, start=1)
        ]


def _to_evidence_item(*, rank: int, hit: VectorSearchResult, vector_name: str) -> EvidenceItem:
    payload = dict(hit.payload)
    text = _payload_text(payload)
    metadata = _evidence_metadata(payload=payload, hit=hit, vector_name=vector_name)

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


def _evidence_metadata(
    *,
    payload: dict[str, Any],
    hit: VectorSearchResult,
    vector_name: str,
) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"text", "contextualized_text"}
    }
    metadata["qdrant_point_id"] = hit.id
    metadata["retrieval_source"] = "vector"
    metadata["retrieval_vector_name"] = vector_name
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
