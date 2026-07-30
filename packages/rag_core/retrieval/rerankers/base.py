from __future__ import annotations

from typing import Protocol, Sequence

from packages.rag_core.retrieval.models import EvidenceItem


class Reranker(Protocol):
    """Interface for query-aware evidence reranking implementations."""

    async def rerank(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        """Return evidence reordered by relevance to the question."""
