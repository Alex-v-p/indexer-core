from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.information_need_graph.models import InformationNeedExecution


class InitializeInformationNeedWorkNode:
    """Create one bounded work item for every decomposed information need."""

    name = "initialize_information_need_work"
    step_type = "orchestration"

    def __init__(self, *, max_attempts_per_information_need: int, max_total_attempts: int) -> None:
        if max_attempts_per_information_need <= 0:
            raise ValueError("max_attempts_per_information_need must be positive.")
        if max_total_attempts <= 0:
            raise ValueError("max_total_attempts must be positive.")
        self._max_attempts = max_attempts_per_information_need
        self._max_total_attempts = max_total_attempts

    async def __call__(self, state: QueryState) -> QueryState:
        decomposition = state.information_need_decomposition
        if decomposition is None:
            raise RuntimeError("Information-need work initialization requires decomposition first.")
        state.information_need_executions = {
            need.need_id: InformationNeedExecution(
                information_need=need,
                max_attempts=self._max_attempts,
            )
            for need in decomposition.information_needs
        }
        state.pending_information_need_ids = [need.need_id for need in decomposition.information_needs]
        state.active_information_need_id = None
        state.information_need_resolution = None
        state.total_information_need_retrieval_attempts = 0
        state.evidence_by_key = {}
        state.next_evidence_rank = 1
        state.primary_document_preference = None
        state.retrieved_evidence = []
        state.evidence_grading = None
        state.retrieval_retry = None
        state.retrieval_plan = None
        state.metadata["information_need_work"] = {
            "information_need_count": len(decomposition.information_needs),
            "max_attempts_per_information_need": self._max_attempts,
            "max_total_retrieval_attempts": self._max_total_attempts,
            "execution_mode": "bounded_queue_with_cyclic_subgraph",
        }
        return state
