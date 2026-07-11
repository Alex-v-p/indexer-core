from __future__ import annotations

import uuid
from typing import Any

from packages.rag_core.providers.keyword_stores import KeywordSearchResult, KeywordStore
from packages.rag_core.retrieval.models import EvidenceItem


class KeywordRetriever:
    """Lexical retriever backed by a keyword-store provider."""

    def __init__(self, *, keyword_store: KeywordStore) -> None:
        self._keyword_store = keyword_store

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        hits = await self._keyword_store.search(question, top_k=top_k)
        return [_to_evidence_item(rank=rank, hit=hit) for rank, hit in enumerate(hits, start=1)]


def _to_evidence_item(*, rank: int, hit: KeywordSearchResult) -> EvidenceItem:
    payload = dict(hit.payload)
    text = _payload_text(payload)
    metadata = {key: value for key, value in payload.items() if key != "text"}
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
