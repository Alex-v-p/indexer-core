from __future__ import annotations

from typing import Any

from packages.rag_core.generation.models import CitationItem
from packages.rag_core.retrieval.models import EvidenceItem


def citation_from_evidence(item: EvidenceItem) -> CitationItem:
    return CitationItem(
        citation_index=item.rank,
        evidence_rank=item.rank,
        label=f"[{item.rank}]",
        page_number=first_int_metadata(item.metadata, "page_number", "source_page_start"),
        quote=item.text[:500],
        qdrant_chunk_index_id=item.qdrant_chunk_index_id,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        metadata={"score": item.score, **item.metadata},
    )


def first_int_metadata(metadata: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None
