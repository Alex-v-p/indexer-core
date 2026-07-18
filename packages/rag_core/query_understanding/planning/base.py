from __future__ import annotations

from typing import Protocol

from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning.models import (
    ClaimPlanningInput,
    ClaimRetrievalPlan,
    RetrievalPlan,
)


class RetrievalPlanner(Protocol):
    """Choose a retrieval strategy from independent query-understanding outputs."""

    async def plan(
        self,
        question: str,
        classification: QueryClassification,
        decomposition: InformationNeedDecomposition,
    ) -> RetrievalPlan:
        """Return the retrieval plan for a classified and decomposed question."""


class ClaimRetrievalPlanner(Protocol):
    """Plan focused lookups for claims that remain unsupported after grading."""

    async def plan_claims(
        self,
        question: str,
        classification: QueryClassification,
        current_plan: RetrievalPlan,
        unresolved_claims: tuple[ClaimPlanningInput, ...],
    ) -> ClaimRetrievalPlan:
        """Return independently executable retrieval tasks for unresolved claims."""
