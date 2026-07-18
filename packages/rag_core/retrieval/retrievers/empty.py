from __future__ import annotations

from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints


class EmptyRetriever:
    """Temporary retriever used until ingestion and vector search are implemented."""

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> list[EvidenceItem]:
        del question, top_k, constraints
        return []
