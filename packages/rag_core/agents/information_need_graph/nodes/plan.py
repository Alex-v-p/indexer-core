from __future__ import annotations

from dataclasses import replace

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.information_need_graph.routes import InformationNeedRoute
from packages.rag_core.query_understanding.planning import (
    InformationNeedPlanningContext,
    InformationNeedPlanningStop,
    InformationNeedRetrievalPlanner,
)


class PlanInformationNeedNode:
    """Create the next complete, claim-specific retrieval attempt."""

    name = "plan_information_need"
    step_type = "information_need_planning"

    def __init__(
        self,
        planner: InformationNeedRetrievalPlanner,
        *,
        available_pipeline_names: tuple[str, ...],
        max_total_attempts: int | None = None,
    ) -> None:
        if max_total_attempts is not None and max_total_attempts <= 0:
            raise ValueError("max_total_attempts must be positive when provided.")
        self._planner = planner
        self._available_pipeline_names = available_pipeline_names
        self._max_total_attempts = max_total_attempts

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None:
            raise RuntimeError("Information-need planning requires an active work item.")
        if execution.classification is None:
            raise RuntimeError("Information-need planning requires classification first.")
        if (
            self._max_total_attempts is not None
            and state.total_information_need_retrieval_attempts >= self._max_total_attempts
        ):
            execution.current_plan = None
            execution.stop_reason = "global_attempt_limit_reached"
            execution.stop_rationale = (
                f"The query-level retrieval budget of {self._max_total_attempts} attempts has been reached."
            )
            execution.next_route = InformationNeedRoute.COMPLETE_EXHAUSTED
            state.metadata["active_information_need_plan"] = {
                "information_need_id": execution.information_need.need_id,
                "reason": execution.stop_reason,
                "rationale": execution.stop_rationale,
            }
            return state

        classification = execution.classification
        parent_classification = state.query_classification
        if (
            not classification.document_constraint.active
            and parent_classification is not None
            and parent_classification.document_constraint.active
        ):
            classification = replace(
                classification,
                document_constraint=parent_classification.document_constraint,
                needs_metadata_filters=True,
                metadata_filter_hints=tuple(
                    dict.fromkeys(
                        (*classification.metadata_filter_hints, *parent_classification.metadata_filter_hints)
                    )
                ),
            )
        if (
            not classification.version_constraint.active
            and parent_classification is not None
            and parent_classification.version_constraint.active
        ):
            classification = replace(
                classification,
                version_constraint=parent_classification.version_constraint,
                needs_metadata_filters=True,
                metadata_filter_hints=tuple(
                    dict.fromkeys(
                        (*classification.metadata_filter_hints, *parent_classification.metadata_filter_hints)
                    )
                ),
            )
        if (
            not classification.date_constraints
            and parent_classification is not None
            and parent_classification.date_constraints
        ):
            classification = replace(
                classification,
                date_constraints=parent_classification.date_constraints,
                needs_metadata_filters=True,
                metadata_filter_hints=tuple(
                    dict.fromkeys(
                        (*classification.metadata_filter_hints, *parent_classification.metadata_filter_hints)
                    )
                ),
            )

        context = InformationNeedPlanningContext(
            original_question=state.question,
            information_need=execution.information_need,
            classification=classification,
            previous_grade=execution.final_grade,
            previous_plans=tuple(execution.plan_history),
            previous_queries=tuple(dict.fromkeys(plan.query for plan in execution.plan_history)),
            available_pipeline_names=self._available_pipeline_names,
            attempts_used=execution.attempts_used,
            max_attempts=execution.max_attempts,
            current_top_k=execution.current_plan.top_k if execution.current_plan is not None else state.top_k,
        )
        result = await self._planner.plan_information_need(context)
        if isinstance(result, InformationNeedPlanningStop):
            execution.current_plan = None
            execution.stop_reason = result.reason
            execution.stop_rationale = result.rationale
            execution.next_route = InformationNeedRoute.COMPLETE_EXHAUSTED
            state.metadata["active_information_need_plan"] = result.to_metadata()
            return state

        execution.current_plan = result
        execution.plan_history.append(result)
        execution.next_route = None
        state.active_retrieval_plan = result.as_retrieval_plan()
        state.active_retrieval_query = result.query
        state.active_retrieval_top_k = result.top_k
        if len(state.information_need_executions) == 1 and state.retrieval_plan is None:
            state.retrieval_plan = result.as_retrieval_plan()
            state.metadata["retrieval_plan"] = state.retrieval_plan.to_metadata()
        state.metadata["active_information_need_plan"] = result.to_metadata()
        return state
