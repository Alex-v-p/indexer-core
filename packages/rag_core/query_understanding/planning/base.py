from __future__ import annotations

from typing import Protocol

from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning.models import (
    ClaimPlanningInput,
    ClaimRetrievalPlan,
    InformationNeedPlanningContext,
    InformationNeedPlanningStop,
    InformationNeedRetrievalPlan,
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
        """Return a compatibility plan for a classified and decomposed question."""


class InformationNeedRetrievalPlanner(Protocol):
    """Create one complete executable plan for one information need."""

    name: str

    async def plan_information_need(
        self,
        context: InformationNeedPlanningContext,
    ) -> InformationNeedRetrievalPlan | InformationNeedPlanningStop:
        """Return the next distinct attempt or an explicit stop decision."""


class ClaimRetrievalPlanner(Protocol):
    """Legacy claim planner retained for compatibility with previous traces."""

    async def plan_claims(
        self,
        question: str,
        classification: QueryClassification,
        current_plan: RetrievalPlan,
        unresolved_claims: tuple[ClaimPlanningInput, ...],
    ) -> ClaimRetrievalPlan:
        """Return independently executable retrieval tasks for unresolved claims."""
