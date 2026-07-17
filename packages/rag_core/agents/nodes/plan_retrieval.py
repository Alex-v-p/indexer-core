from __future__ import annotations

from packages.rag_core.agents.state import QueryState
from packages.rag_core.query_understanding.planning import RetrievalPlanner


class PlanRetrievalNode:
    """Agent node that turns query classification into an executable retrieval plan."""

    name = "plan_retrieval"
    step_type = "planning"

    def __init__(self, planner: RetrievalPlanner) -> None:
        self._planner = planner

    async def __call__(self, state: QueryState) -> QueryState:
        if state.query_classification is None:
            raise RuntimeError("Retrieval planning requires query classification to run first.")

        plan = await self._planner.plan(state.question, state.query_classification)
        state.retrieval_plan = plan
        state.metadata["retrieval_plan"] = plan.to_metadata()
        return state
