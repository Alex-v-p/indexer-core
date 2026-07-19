from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.information_need_graph.models import InformationNeedExecution
from packages.rag_core.agents.information_need_graph.reporting import InformationNeedResolutionReport
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)


class AggregateInformationNeedsNode:
    """Combine completed work items into final grading and answer evidence."""

    name = "aggregate_information_needs"
    step_type = "aggregation"

    def __init__(
        self,
        *,
        subgraph_name: str,
        max_total_attempts: int,
        max_attempts_per_information_need: int,
    ) -> None:
        self._subgraph_name = subgraph_name
        self._max_total_attempts = max_total_attempts
        self._max_attempts_per_need = max_attempts_per_information_need

    async def __call__(self, state: QueryState) -> QueryState:
        executions = tuple(state.information_need_executions.values())
        if not executions:
            raise RuntimeError("Information-need aggregation requires initialized work items.")
        information_need_grades = tuple(
            execution.final_grade or _missing_grade_for_unattempted_execution(execution)
            for execution in executions
        )
        grade_by_rank: dict[int, list[EvidenceGrade]] = {}
        fallback_used = False
        for execution in executions:
            if execution.last_grading is None:
                continue
            fallback_used = fallback_used or execution.last_grading.fallback_used
            for grade in execution.last_grading.grades:
                grade_by_rank.setdefault(grade.evidence_rank, []).append(grade)

        evidence_grades: list[EvidenceGrade] = []
        for item in sorted(state.retrieved_evidence, key=lambda candidate: candidate.rank):
            source_grades = grade_by_rank.get(item.rank, [])
            relevant_source_grades = tuple(grade for grade in source_grades if grade.relevant)
            supports = tuple(
                dict.fromkeys(
                    need_id
                    for grade in relevant_source_grades
                    for need_id in grade.supports_information_need_ids
                ),
            )
            relevant = bool(relevant_source_grades and supports)
            evidence_grades.append(
                EvidenceGrade(
                    evidence_rank=item.rank,
                    relevance_score=max((grade.relevance_score for grade in source_grades), default=0.0),
                    relevant=relevant,
                    rationale=(
                        "Retained because at least one per-information-need grader directly approved this chunk."
                        if relevant
                        else "Rejected because no per-information-need grader directly approved this chunk."
                    ),
                    supports_information_need_ids=supports if relevant else (),
                ),
            )

        required_grades = tuple(grade for grade in information_need_grades if grade.required)
        relevant_evidence_count = sum(1 for grade in evidence_grades if grade.relevant)
        if relevant_evidence_count and required_grades and all(grade.supported for grade in required_grades):
            status = EvidenceSufficiency.SUFFICIENT
        elif relevant_evidence_count:
            status = EvidenceSufficiency.WEAK
        else:
            status = EvidenceSufficiency.MISSING
        coverage = (
            sum(grade.coverage_score for grade in required_grades) / len(required_grades)
            if required_grades
            else 1.0
        )
        report = EvidenceGradingReport(
            status=status,
            coverage_score=round(coverage, 4),
            grades=tuple(evidence_grades),
            information_need_grades=information_need_grades,
            rationale=(
                "Aggregated the final independent grade for every information need after its bounded subgraph lifecycle."
            ),
            grader_name="hierarchical_information_need_aggregation",
            fallback_used=fallback_used,
        )
        resolution = InformationNeedResolutionReport(
            graph_name=self._subgraph_name,
            executions=executions,
            total_retrieval_attempts=state.total_information_need_retrieval_attempts,
            max_total_retrieval_attempts=self._max_total_attempts,
            max_attempts_per_information_need=self._max_attempts_per_need,
        )
        state.evidence_grading = report
        state.information_need_resolution = resolution
        state.metadata["evidence_grading"] = report.to_metadata()
        state.metadata["information_need_resolution"] = resolution.to_metadata()
        state.metadata["information_need_executions"] = [execution.to_metadata() for execution in executions]
        return state


def _missing_grade_for_unattempted_execution(execution: InformationNeedExecution) -> InformationNeedGrade:
    need = execution.information_need
    return InformationNeedGrade(
        information_need_id=need.need_id,
        description=need.description,
        status=InformationNeedSupport.MISSING,
        coverage_score=0.0,
        supporting_evidence_ranks=(),
        rationale=execution.stop_rationale or "No retrieval attempt was available for this information need.",
        required=need.required,
    )
