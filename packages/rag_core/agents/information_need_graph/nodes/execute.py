from __future__ import annotations

from packages.rag_core.agents.information_need_graph.evidence import (
    evidence_key,
    merge_information_need_evidence,
)
from packages.rag_core.agents.information_need_graph.models import (
    DocumentBalancingMetadata,
    InformationNeedAttemptEvidence,
    RerankingMetadata,
    RetrievalExecutionMetadata,
)
from packages.rag_core.agents.shared.retrieval.executor import RetrievalPlanExecutor
from packages.rag_core.agents.query_graph.state import QueryState


class ExecuteInformationNeedPlanNode:
    """Execute only the active information item's current plan."""

    name = "execute_information_need_plan"
    step_type = "information_need_retrieval"

    def __init__(
        self,
        retrieval_executor: RetrievalPlanExecutor,
        *,
        max_total_attempts: int,
        max_accumulated_evidence: int,
    ) -> None:
        if max_total_attempts <= 0 or max_accumulated_evidence <= 0:
            raise ValueError("Attempt and evidence limits must be positive.")
        self._retrieval_executor = retrieval_executor
        self._max_total_attempts = max_total_attempts
        self._max_accumulated_evidence = max_accumulated_evidence

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None or execution.current_plan is None:
            raise RuntimeError("Information-need retrieval requires an active plan.")
        if state.total_information_need_retrieval_attempts >= self._max_total_attempts:
            execution.stop_reason = "global_attempt_limit_reached"
            execution.stop_rationale = (
                f"The query-level retrieval budget of {self._max_total_attempts} attempts was reached before execution."
            )
            raise RuntimeError(execution.stop_rationale)

        plan = execution.current_plan
        retrieved, execution_metadata = await self._retrieval_executor.execute_lookup(
            plan=plan.as_retrieval_plan(),
            query=plan.query,
            top_k=plan.top_k,
        )
        keys, unique_added = merge_information_need_evidence(
            state,
            retrieved,
            information_need_id=execution.information_need.need_id,
            attempt_number=plan.attempt_number,
            query=plan.query,
            max_items=self._max_accumulated_evidence,
        )
        for key in keys:
            execution.add_evidence_key(key)
        coverage = execution_metadata.get("coverage")
        enforcement = state.metadata.get("document_scope_enforcement")
        if isinstance(coverage, dict) and isinstance(enforcement, dict):
            coverage["out_of_scope_rejected_count"] = max(
                int(coverage.get("out_of_scope_rejected_count", 0)),
                int(enforcement.get("out_of_scope_rejected_count", 0)),
            )
        execution.pending_attempt_evidence = tuple(
            InformationNeedAttemptEvidence.capture(
                item,
                evidence_key=(key := evidence_key(item)),
                retrieval_order=index,
                aggregate_rank=(
                    state.evidence_by_key[key].rank if key in state.evidence_by_key else None
                ),
            )
            for index, item in enumerate(retrieved, start=1)
        )
        execution.pending_retrieval_metadata = RetrievalExecutionMetadata.from_execution(
            execution_metadata,
            retrieved,
        )
        execution.pending_reranking_metadata = RerankingMetadata.from_execution(execution_metadata)
        execution.pending_document_balancing = DocumentBalancingMetadata.from_execution(
            execution_metadata,
        )
        state.total_information_need_retrieval_attempts += 1
        state.metadata["active_information_need_lookup"] = {
            "information_need_id": execution.information_need.need_id,
            "attempt_number": plan.attempt_number,
            "retrieved_count": len(retrieved),
            "unique_evidence_added": unique_added,
            "evidence_keys": list(keys),
            "evidence": [
                item.to_metadata() for item in execution.pending_attempt_evidence
            ],
            "retrieval_metadata": execution.pending_retrieval_metadata.to_metadata(),
            "reranking_metadata": execution.pending_reranking_metadata.to_metadata(),
            "document_balancing_metadata": execution.pending_document_balancing.to_metadata(),
            "execution": execution_metadata,
            "primary_document": (
                plan.preferred_document.document.display_name
                if plan.preferred_document is not None
                else None
            ),
            "document_balancing": execution_metadata.get("document_balancing", {}),
            "subject_lane": (
                execution.information_need.subject_lane.to_metadata()
                if execution.information_need.subject_lane is not None
                else None
            ),
            "coverage": execution_metadata.get("coverage", {}),
        }
        return state
