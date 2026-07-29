from __future__ import annotations

from packages.rag_core.agents.information_need_graph.evidence import evidence_for_information_need, prune_information_need_evidence
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.information_need_graph.models import InformationNeedAttempt
from packages.rag_core.retrieval.graders import EvidenceGrader
from packages.rag_core.retrieval.models import RetrievalConstraints
from packages.rag_core.retrieval.retrievers.base import callable_accepts_parameter


class GradeInformationNeedNode:
    """Grade one information item against only its accumulated evidence."""

    name = "grade_information_need"
    step_type = "information_need_evidence_grading"

    def __init__(self, grader: EvidenceGrader) -> None:
        self._grader = grader

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None or execution.current_plan is None:
            raise RuntimeError("Information-need grading requires an active retrieval plan.")
        evidence = evidence_for_information_need(state, execution.information_need.need_id)
        grade_information_needs = getattr(self._grader, "grade_information_needs", None)
        constraints = RetrievalConstraints(
            document=execution.current_plan.document_constraint,
            version=execution.current_plan.version_constraint,
            dates=execution.current_plan.date_constraints,
        )
        if callable(grade_information_needs):
            if callable_accepts_parameter(grade_information_needs, "constraints"):
                report = await grade_information_needs(
                    state.question,
                    evidence,
                    (execution.information_need,),
                    constraints=constraints,
                )
            else:
                report = await grade_information_needs(
                    state.question,
                    evidence,
                    (execution.information_need,),
                )
        else:
            grade = self._grader.grade
            if callable_accepts_parameter(grade, "constraints"):
                report = await grade(
                    execution.information_need.retrieval_query,
                    evidence,
                    constraints=constraints,
                )
            else:
                report = await grade(execution.information_need.retrieval_query, evidence)
        if len(report.information_need_grades) != 1:
            raise RuntimeError("Per-information-need grading must return exactly one information-need grade.")

        execution.last_grading = report
        execution.final_grade = report.information_need_grades[0]
        lookup = state.metadata.get("active_information_need_lookup", {})
        retrieved_count = int(lookup.get("retrieved_count", len(evidence))) if isinstance(lookup, dict) else len(evidence)
        unique_added = int(lookup.get("unique_evidence_added", 0)) if isinstance(lookup, dict) else 0
        prune_information_need_evidence(
            state,
            information_need_id=execution.information_need.need_id,
            relevant_ranks=report.relevant_evidence_ranks,
        )
        grades_by_rank = {grade.evidence_rank: grade for grade in report.grades}
        retained_keys = set(execution.evidence_keys)
        attempt_evidence = tuple(
            item.with_grading(
                grades_by_rank.get(item.aggregate_rank),
                retained_after_need_grading=item.evidence_key in retained_keys,
            )
            for item in execution.pending_attempt_evidence
        )
        execution.attempts.append(
            InformationNeedAttempt(
                attempt_number=execution.current_plan.attempt_number,
                plan=execution.current_plan,
                grading=report,
                retrieved_count=retrieved_count,
                unique_evidence_added=unique_added,
                evidence_keys=tuple(execution.evidence_keys),
                constraint_validation=_require_constraint_validation(execution),
                evidence=attempt_evidence,
                retrieval_metadata=execution.pending_retrieval_metadata,
                reranking_metadata=execution.pending_reranking_metadata,
                document_balancing=execution.pending_document_balancing,
            ),
        )
        execution.pending_attempt_evidence = ()
        execution.pending_retrieval_metadata = type(execution.pending_retrieval_metadata)()
        execution.pending_reranking_metadata = type(execution.pending_reranking_metadata)()
        execution.pending_document_balancing = type(execution.pending_document_balancing)()
        state.metadata["active_information_need_grading"] = {
            "information_need_id": execution.information_need.need_id,
            "attempt_number": execution.current_plan.attempt_number,
            "report": report.to_metadata(),
        }
        return state


def _require_constraint_validation(execution):
    report = execution.last_constraint_validation
    if report is None:
        raise RuntimeError("Evidence grading requires constraint validation first.")
    return report
