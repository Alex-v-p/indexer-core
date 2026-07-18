from __future__ import annotations

from packages.rag_core.agents.state import QueryState
from packages.rag_core.query_understanding.planning import RetrievalPlanner


class PlanRetrievalNode:
    """Turn independent query-understanding outputs into a retrieval decision."""

    name = "plan_retrieval"
    step_type = "planning"

    def __init__(self, planner: RetrievalPlanner) -> None:
        self._planner = planner

    async def __call__(self, state: QueryState) -> QueryState:
        if state.query_classification is None:
            raise RuntimeError("Retrieval planning requires query classification to run first.")
        if state.information_need_decomposition is None:
            raise RuntimeError("Retrieval planning requires information-need decomposition to run first.")

        plan = await self._planner.plan(
            state.question,
            state.query_classification,
            state.information_need_decomposition,
        )
        state.retrieval_plan = plan
        state.active_retrieval_plan = plan
        state.active_retrieval_query = state.question
        state.active_retrieval_top_k = state.top_k
        state.metadata["retrieval_plan"] = plan.to_metadata()
        return state
