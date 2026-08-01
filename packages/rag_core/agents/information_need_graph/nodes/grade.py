from __future__ import annotations

from dataclasses import replace

from packages.rag_core.agents.information_need_graph.evidence import (
    evidence_for_information_need,
    prune_information_need_evidence,
)
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.information_need_graph.models import InformationNeedAttempt
from packages.rag_core.document_scope import CoverageMode
from packages.rag_core.retrieval.graders import EvidenceGrader
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
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
            document_scope=execution.current_plan.document_scope,
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
        report, coverage_retained_ranks = (
            preserve_information_need_multi_document_coverage(
                state,
                evidence=evidence,
                report=report,
            )
        )

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
        retained_evidence = evidence_for_information_need(
            state,
            execution.information_need.need_id,
        )
        contributing_ids = tuple(
            dict.fromkeys(
                str(item.document_id)
                for item in retained_evidence
                if item.document_id is not None
            )
        )
        coverage = dict(execution.pending_retrieval_metadata.coverage)
        searched_count = int(coverage.get("searched_document_count", 0))
        effective_mode = str(coverage.get("effective_mode", "best_evidence"))
        coverage["contributing_document_count"] = len(contributing_ids)
        coverage["contributing_document_ids"] = list(contributing_ids[:20])
        coverage["coverage_satisfied"] = _coverage_satisfied(
            effective_mode=effective_mode,
            searched_document_count=searched_count,
            contributing_document_count=len(contributing_ids),
        )
        if coverage_retained_ranks:
            coverage["coverage_retained_ranks"] = list(coverage_retained_ranks)
            coverage["coverage_retention_policy"] = (
                "best_constraint_matched_candidate_per_searched_document"
            )
        execution.pending_retrieval_metadata = replace(
            execution.pending_retrieval_metadata,
            coverage=coverage,
        )
        if isinstance(lookup, dict):
            raw_execution = lookup.get("execution")
            if isinstance(raw_execution, dict):
                raw_execution["coverage"] = coverage
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


def preserve_information_need_multi_document_coverage(
    state: QueryState,
    *,
    evidence: list[EvidenceItem],
    report: EvidenceGradingReport,
) -> tuple[EvidenceGradingReport, tuple[int, ...]]:
    """Keep one safe constraint-matched candidate per independently searched document."""

    execution = state.active_information_need_execution
    if execution is None or execution.current_plan is None:
        return report, ()
    plan = execution.current_plan
    coverage = execution.pending_retrieval_metadata.coverage
    searched_document_ids = {
        str(item) for item in coverage.get("searched_document_ids", [])
    }
    if (
        plan.coverage_mode is not CoverageMode.MULTI_DOCUMENT
        or coverage.get("effective_mode") != CoverageMode.MULTI_DOCUMENT.value
        or not plan.document_scope.strict
        or len(plan.document_scope.allowed_document_ids) < 2
        or len(searched_document_ids) < 2
    ):
        return report, ()

    grades_by_rank = {grade.evidence_rank: grade for grade in report.grades}
    need_id = execution.information_need.need_id
    original_need = report.information_need_grades[0]
    relevant_documents = {
        item.document_id
        for item in evidence
        if (grade := grades_by_rank.get(item.rank)) is not None
        and grade.relevant
        and need_id in grade.supports_information_need_ids
        and item.rank in original_need.supporting_evidence_ranks
        and item.text.strip()
        and _plan_scope_safe(plan, item)
    }
    if (
        original_need.status is InformationNeedSupport.MISSING
        or not relevant_documents
    ):
        return report, ()
    candidates_by_document: dict[object, list[EvidenceItem]] = {}
    for item in sorted(evidence, key=lambda candidate: candidate.rank):
        if (
            item.document_id is None
            or str(item.document_id) not in searched_document_ids
            or item.rank not in grades_by_rank
            or not item.text.strip()
            or not _plan_scope_safe(plan, item)
        ):
            continue
        candidates_by_document.setdefault(item.document_id, []).append(item)

    retained: list[int] = []
    for document_id, candidates in candidates_by_document.items():
        if document_id in relevant_documents:
            continue
        selected = candidates[0]
        prior_grade = grades_by_rank[selected.rank]
        grades_by_rank[selected.rank] = EvidenceGrade(
            evidence_rank=selected.rank,
            relevance_score=prior_grade.relevance_score,
            relevant=True,
            rationale=(
                "Retained for multi-document coverage as the highest-ranked non-empty "
                "constraint-matched candidate from an independently searched in-scope document."
            ),
            supports_information_need_ids=(need_id,),
        )
        retained.append(selected.rank)
        relevant_documents.add(document_id)

    if not retained:
        return report, ()

    supporting = tuple(
        dict.fromkeys((*original_need.supporting_evidence_ranks, *retained)),
    )
    need_grade = InformationNeedGrade(
        information_need_id=original_need.information_need_id,
        description=original_need.description,
        status=original_need.status,
        coverage_score=original_need.coverage_score,
        supporting_evidence_ranks=supporting,
        rationale=original_need.rationale,
        required=original_need.required,
    )
    relevant_count = sum(grade.relevant for grade in grades_by_rank.values())
    status = (
        EvidenceSufficiency.MISSING
        if relevant_count == 0
        else EvidenceSufficiency.SUFFICIENT
        if not need_grade.required or need_grade.supported
        else EvidenceSufficiency.WEAK
    )
    return (
        EvidenceGradingReport(
            status=status,
            coverage_score=need_grade.coverage_score if need_grade.required else 1.0,
            grades=tuple(grades_by_rank[rank] for rank in sorted(grades_by_rank)),
            information_need_grades=(need_grade,),
            rationale=(
                f"{report.rationale} Multi-document coverage retained ranks "
                f"{sorted(retained)} before need-level pruning."
            ),
            grader_name=report.grader_name,
            fallback_used=report.fallback_used,
            structured_output=report.structured_output,
        ),
        tuple(sorted(retained)),
    )


def _plan_scope_safe(plan, item: EvidenceItem) -> bool:
    if not plan.document_scope.allows(item.document_id):
        return False
    lane = plan.subject_lane
    if lane is None:
        return True
    return bool(
        item.subject_lane_id == lane.lane_id
        and item.subject_id == lane.subject_id
        and item.subject_name == lane.subject_name
        and lane.document_scope.allows(item.document_id)
    )


def _require_constraint_validation(execution):
    report = execution.last_constraint_validation
    if report is None:
        raise RuntimeError("Evidence grading requires constraint validation first.")
    return report


def _coverage_satisfied(
    *,
    effective_mode: str,
    searched_document_count: int,
    contributing_document_count: int,
) -> bool:
    if effective_mode == "best_evidence":
        return contributing_document_count > 0
    return (
        searched_document_count > 0
        and contributing_document_count >= min(2, searched_document_count)
    )
