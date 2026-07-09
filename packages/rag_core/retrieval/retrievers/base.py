from __future__ import annotations

from typing import Protocol

from packages.rag_core.retrieval.models import EvidenceItem


class Retriever(Protocol):
    """Interface for query-time retrieval implementations."""

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        """Return ranked evidence for a question."""
