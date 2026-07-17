from __future__ import annotations

from typing import Protocol

from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.planning.models import RetrievalPlan


class RetrievalPlanner(Protocol):
    """Choose a retrieval strategy from a structured query classification."""

    async def plan(self, question: str, classification: QueryClassification) -> RetrievalPlan:
        """Return the retrieval plan for a non-empty question."""
