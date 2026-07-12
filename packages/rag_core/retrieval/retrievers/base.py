from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from packages.rag_core.retrieval.models import EvidenceItem


@dataclass(slots=True)
class RetrievalBatch:
    """Evidence plus strategy-specific metadata for graph traces and evaluation."""

    evidence: list[EvidenceItem]
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Interface for query-time retrieval implementations."""

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        """Return ranked evidence for a question."""
