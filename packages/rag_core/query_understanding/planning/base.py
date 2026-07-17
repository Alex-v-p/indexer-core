from __future__ import annotations

from typing import Protocol

from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning.models import RetrievalPlan


class RetrievalPlanner(Protocol):
    """Choose a retrieval strategy from independent query-understanding outputs."""

    async def plan(
        self,
        question: str,
        classification: QueryClassification,
        decomposition: InformationNeedDecomposition,
    ) -> RetrievalPlan:
        """Return the retrieval plan for a classified and decomposed question."""
