from __future__ import annotations

from typing import Protocol

from packages.rag_core.agents.classification.models import QueryClassification


class QueryClassifier(Protocol):
    """Classify a question before retrieval planning and execution."""

    async def classify(self, question: str) -> QueryClassification:
        """Return a structured classification for a non-empty user question."""
