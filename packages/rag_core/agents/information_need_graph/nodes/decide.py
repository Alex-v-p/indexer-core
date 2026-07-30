from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.information_need_graph.routes import InformationNeedRoute
from packages.rag_core.retrieval.retry import (
    InformationNeedRetryAction,
    InformationNeedRetryContext,
    RetrievalRetryPolicy,
)


class DecideInformationNeedNode:
    """Route the active item after grading while enforcing bounded retries."""

    name = "decide_information_need"
    step_type = "information_need_control"

    def __init__(
        self,
        policy: RetrievalRetryPolicy,
        *,
        max_total_attempts: int,
        max_reclassifications: int = 1,
    ) -> None:
        if max_total_attempts <= 0 or max_reclassifications < 0:
            raise ValueError("Controller limits are invalid.")
        self._policy = policy
        self._max_total_attempts = max_total_attempts
        self._max_reclassifications = max_reclassifications

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None or execution.last_grading is None or execution.classification is None:
            raise RuntimeError("Information-need decision requires classification and grading.")
        decision = self._policy.decide_information_need(
            InformationNeedRetryContext(
                information_need_id=execution.information_need.need_id,
                evidence_grading=execution.last_grading,
                attempts_used=execution.attempts_used,
                max_attempts=execution.max_attempts,
                total_attempts_used=state.total_information_need_retrieval_attempts,
                max_total_attempts=self._max_total_attempts,
                classification_confidence=execution.classification.confidence,
                reclassifications_used=execution.reclassifications_used,
                max_reclassifications=self._max_reclassifications,
            ),
        )
        route_map = {
            InformationNeedRetryAction.RETRY: InformationNeedRoute.RETRY,
            InformationNeedRetryAction.RECLASSIFY: InformationNeedRoute.RECLASSIFY,
            InformationNeedRetryAction.COMPLETE_SUPPORTED: InformationNeedRoute.COMPLETE_SUPPORTED,
            InformationNeedRetryAction.COMPLETE_EXHAUSTED: InformationNeedRoute.COMPLETE_EXHAUSTED,
        }
        execution.next_route = route_map[decision.action]
        if decision.action in {
            InformationNeedRetryAction.COMPLETE_SUPPORTED,
            InformationNeedRetryAction.COMPLETE_EXHAUSTED,
        }:
            execution.stop_reason = decision.reason
            execution.stop_rationale = decision.rationale
        state.metadata["active_information_need_decision"] = {
            "information_need_id": execution.information_need.need_id,
            **decision.to_metadata(),
        }
        return state
