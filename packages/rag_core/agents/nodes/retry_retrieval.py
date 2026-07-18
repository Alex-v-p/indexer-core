from __future__ import annotations

from packages.rag_core.agents.nodes.execute_retrieval_plan import ExecuteRetrievalPlanNode
from packages.rag_core.agents.nodes.grade_evidence import GradeEvidenceNode
from packages.rag_core.agents.state import QueryState
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.graders import InformationNeedSupport
from packages.rag_core.retrieval.retry import (
    RetryAction,
    RetrievalAttempt,
    RetrievalRetryContext,
    RetrievalRetryPolicy,
    RetrievalRetryReport,
)


class RetryRetrievalNode:
    """Retry weak retrieval through bounded query, top-k, and pipeline fallbacks."""

    name = "retry_retrieval"
    step_type = "retrieval_retry"

    def __init__(
        self,
        *,
        retry_policy: RetrievalRetryPolicy,
        retrieval_node: ExecuteRetrievalPlanNode,
        grading_node: GradeEvidenceNode,
    ) -> None:
        self._retry_policy = retry_policy
        self._retrieval_node = retrieval_node
        self._grading_node = grading_node

    async def __call__(self, state: QueryState) -> QueryState:
        plan = state.effective_retrieval_plan
        grading = state.evidence_grading
        if plan is None:
            raise RuntimeError("Retry evaluation requires a retrieval plan.")
        if grading is None:
            raise RuntimeError("Retry evaluation requires evidence grading.")

        attempts = [self._snapshot_attempt(state, retry_number=0)]
        stop_reason = None
        stop_rationale = None

        while True:
            current_plan = state.effective_retrieval_plan
            current_grading = state.evidence_grading
            if current_plan is None or current_grading is None:
                raise RuntimeError("Retry state lost its active retrieval plan or evidence grading report.")

            context = RetrievalRetryContext(
                original_question=state.question,
                current_query=state.effective_retrieval_query,
                current_top_k=state.effective_retrieval_top_k,
                current_plan=current_plan,
                evidence_grading=current_grading,
                retries_used=len(attempts) - 1,
                attempted_strategies=tuple(attempt.strategy for attempt in attempts),
                unresolved_retrieval_queries=self._unresolved_retrieval_queries(state),
                available_pipeline_names=self._retrieval_node.available_pipeline_names,
            )
            decision = self._retry_policy.decide(context)
            if not decision.should_retry:
                stop_reason = decision.stop_reason
                stop_rationale = decision.rationale
                break

            state.active_retrieval_plan = decision.next_plan
            state.active_retrieval_query = decision.next_query
            state.active_retrieval_top_k = decision.next_top_k
            state.evidence_grading = None
            state = await self._retrieval_node(state)
            state = await self._grading_node(state)
            attempts.append(
                self._snapshot_attempt(
                    state,
                    retry_number=len(attempts),
                    actions=decision.actions,
                    decision_rationale=decision.rationale,
                ),
            )

        if stop_reason is None or stop_rationale is None:
            raise RuntimeError("Retry policy stopped without a reason and rationale.")

        report = RetrievalRetryReport(
            policy_name=self._retry_policy.name,
            max_retries=self._retry_policy.max_retries,
            attempts=tuple(attempts),
            stop_reason=stop_reason,
            stop_rationale=stop_rationale,
        )
        state.retrieval_retry = report
        state.metadata["retrieval_retry"] = report.to_metadata()
        final_plan = state.effective_retrieval_plan
        if final_plan is not None:
            state.metadata["final_retrieval_plan"] = final_plan.to_metadata()
        return state

    @staticmethod
    def _snapshot_attempt(
        state: QueryState,
        *,
        retry_number: int,
        actions: tuple[RetryAction, ...] = (),
        decision_rationale: str | None = None,
    ) -> RetrievalAttempt:
        plan = state.effective_retrieval_plan
        grading = state.evidence_grading
        if plan is None or grading is None:
            raise RuntimeError("Cannot snapshot a retrieval attempt before plan execution and grading.")
        return RetrievalAttempt(
            attempt_number=retry_number + 1,
            retry_number=retry_number,
            query=state.effective_retrieval_query,
            top_k=state.effective_retrieval_top_k,
            retrieval_plan=plan,
            evidence_grading=grading,
            evidence_count=len(state.retrieved_evidence),
            actions=actions,
            decision_rationale=decision_rationale,
        )

    @staticmethod
    def _unresolved_retrieval_queries(state: QueryState) -> tuple[str, ...]:
        decomposition = state.information_need_decomposition
        grading = state.evidence_grading
        if decomposition is None or grading is None:
            return ()

        unresolved_ids = {
            grade.information_need_id
            for grade in grading.information_need_grades
            if grade.required and grade.status is not InformationNeedSupport.SUPPORTED
        }
        if not unresolved_ids:
            unresolved_descriptions = set(grading.unresolved_information)
            needs = tuple(
                need
                for need in decomposition.information_needs
                if need.required and need.description in unresolved_descriptions
            )
        else:
            needs = tuple(
                need
                for need in decomposition.information_needs
                if need.required and need.need_id in unresolved_ids
            )
        return _unique_queries(needs)


def _unique_queries(needs: tuple[InformationNeed, ...]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for need in needs:
        query = " ".join(need.retrieval_query.strip().split())
        key = query.lower()
        if not query or key in seen:
            continue
        values.append(query)
        seen.add(key)
    return tuple(values)
