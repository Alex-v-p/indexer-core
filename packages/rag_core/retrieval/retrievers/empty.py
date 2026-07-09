from __future__ import annotations

from packages.rag_core.retrieval.models import EvidenceItem


class EmptyRetriever:
    """Temporary retriever used until ingestion and vector search are implemented."""

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        return []
