from __future__ import annotations

from packages.rag_core.agents.state import QueryState
from packages.rag_core.agents.work_items import (
    InformationNeedExecutionStatus,
    InformationNeedRoute,
)


class CompleteInformationNeedNode:
    """Finalize one work item and release the subgraph to select another."""

    name = "complete_information_need"
    step_type = "orchestration"

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None:
            raise RuntimeError("Completion requires an active information need.")
        if execution.next_route is InformationNeedRoute.COMPLETE_SUPPORTED:
            execution.status = InformationNeedExecutionStatus.SUPPORTED
        elif execution.next_route is InformationNeedRoute.COMPLETE_EXHAUSTED:
            execution.status = InformationNeedExecutionStatus.EXHAUSTED
        else:
            raise RuntimeError("Information need reached completion without a terminal route.")

        state.metadata["last_completed_information_need"] = execution.to_metadata()
        state.active_information_need_id = None
        state.active_retrieval_plan = None
        state.active_retrieval_query = None
        state.active_retrieval_top_k = None
        return state
