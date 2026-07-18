from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.information_need_graph.routes import InformationNeedExecutionStatus


class SelectInformationNeedNode:
    """Activate the next pending work item from the bounded queue."""

    name = "select_information_need"
    step_type = "orchestration"

    async def __call__(self, state: QueryState) -> QueryState:
        if state.active_information_need_id is not None:
            return state
        while state.pending_information_need_ids:
            need_id = state.pending_information_need_ids.pop(0)
            execution = state.information_need_executions[need_id]
            if execution.status is not InformationNeedExecutionStatus.PENDING:
                continue
            execution.status = InformationNeedExecutionStatus.ACTIVE
            execution.next_route = None
            state.active_information_need_id = need_id
            return state
        return state
