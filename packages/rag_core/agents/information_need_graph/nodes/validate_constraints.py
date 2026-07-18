from __future__ import annotations

from packages.rag_core.agents.information_need_graph.evidence import (
    evidence_for_information_need,
    prune_information_need_evidence,
)
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.retrieval.constraint_validation import validate_evidence_constraints
from packages.rag_core.retrieval.models import RetrievalConstraints


class ValidateInformationNeedConstraintsNode:
    """Reject evidence outside an information need's planned metadata scope."""

    name = "validate_information_need_constraints"
    step_type = "constraint_validation"

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None or execution.current_plan is None:
            raise RuntimeError("Constraint validation requires an active information-need plan.")

        plan = execution.current_plan
        constraints = RetrievalConstraints(
            document=plan.document_constraint,
            version=plan.version_constraint,
            dates=plan.date_constraints,
        )
        evidence = evidence_for_information_need(state, execution.information_need.need_id)
        matched, report = validate_evidence_constraints(evidence, constraints)
        prune_information_need_evidence(
            state,
            information_need_id=execution.information_need.need_id,
            relevant_ranks=tuple(item.rank for item in matched),
        )
        execution.last_constraint_validation = report
        execution.constraint_validation_history.append(report)
        state.metadata["active_information_need_constraint_validation"] = {
            "information_need_id": execution.information_need.need_id,
            "attempt_number": plan.attempt_number,
            **report.to_metadata(),
        }
        return state
