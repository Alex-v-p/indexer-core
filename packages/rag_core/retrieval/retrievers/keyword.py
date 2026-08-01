from __future__ import annotations

import uuid
from typing import Any

from packages.rag_core.ports import KeywordSearchResult, KeywordStore
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
from packages.rag_core.retrieval.retrievers.base import callable_accepts_parameter
from packages.rag_core.retrieval.retrievers.base import (
    RetrievalBatch,
    UnsupportedStrictDocumentScopeError,
    enforce_document_scope_with_count,
)


class KeywordRetriever:
    """Lexical retriever backed by a keyword-store provider."""

    def __init__(self, *, keyword_store: KeywordStore) -> None:
        self._keyword_store = keyword_store

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> list[EvidenceItem]:
        return (
            await self.retrieve_with_metadata(
                question,
                top_k=top_k,
                constraints=constraints,
            )
        ).evidence

    async def retrieve_with_metadata(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> RetrievalBatch:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if constraints is not None and constraints.document_scope.is_strict_empty:
            return RetrievalBatch(
                evidence=[],
                metadata={"out_of_scope_rejected_count": 0},
            )

        search = self._keyword_store.search
        search_kwargs = {"top_k": top_k}
        if constraints is not None and callable_accepts_parameter(search, "document_constraint"):
            search_kwargs["document_constraint"] = constraints.document
        if constraints is not None and callable_accepts_parameter(search, "version_constraint"):
            search_kwargs["version_constraint"] = constraints.version
        if constraints is not None and callable_accepts_parameter(search, "date_constraints"):
            search_kwargs["date_constraints"] = constraints.dates
        if constraints is not None and callable_accepts_parameter(search, "document_scope"):
            search_kwargs["document_scope"] = constraints.document_scope
        elif constraints is not None and constraints.document_scope.strict:
            raise UnsupportedStrictDocumentScopeError(
                f"{type(self._keyword_store).__name__} cannot enforce a strict document scope.",
            )
        hits = await search(question, **search_kwargs)
        evidence, rejected = enforce_document_scope_with_count(
            [_to_evidence_item(rank=rank, hit=hit) for rank, hit in enumerate(hits, start=1)],
            constraints,
        )
        return RetrievalBatch(
            evidence=evidence,
            metadata={
                "strategy": "keyword",
                "out_of_scope_rejected_count": rejected,
            },
        )


def _to_evidence_item(*, rank: int, hit: KeywordSearchResult) -> EvidenceItem:
    payload = dict(hit.payload)
    text = _payload_text(payload)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"text", "contextualized_text"}
    }
    metadata["keyword_store_document_id"] = hit.id
    metadata["retrieval_source"] = "keyword"
    metadata["score"] = hit.score

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


def _optional_uuid(value: object) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
