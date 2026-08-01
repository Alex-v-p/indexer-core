from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.document_scope import CoverageMode
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.arbitration import EvidenceArbitrator
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints


class ArbitrateFinalEvidenceNode:
    """Apply a strict original-question-level filter after per-need aggregation."""

    name = "arbitrate_final_evidence"
    step_type = "evidence_arbitration"

    def __init__(self, arbitrator: EvidenceArbitrator) -> None:
        self._arbitrator = arbitrator

    async def __call__(self, state: QueryState) -> QueryState:
        prior = state.evidence_grading
        if prior is None:
            raise RuntimeError("Final evidence arbitration requires an aggregated grading report.")
        information_needs = _information_needs(state)
        report = await self._arbitrator.arbitrate(
            state.question,
            state.retrieved_evidence,
            information_needs,
            prior,
            constraints=_query_constraints(state),
        )
        report, restored_ranks = preserve_multi_document_coverage(
            state,
            prior=prior,
            final=report,
        )
        grades_by_rank = {grade.evidence_rank: grade for grade in report.grades}
        for item in state.retrieved_evidence:
            grade = grades_by_rank.get(item.rank)
            if grade is not None:
                item.metadata["final_evidence_arbitration"] = grade.to_metadata()

        state.evidence_grading = report
        state.metadata["evidence_arbitration"] = report.to_metadata()
        state.metadata["evidence_grading"] = report.to_metadata()
        state.metadata["unresolved_information"] = list(report.unresolved_information)
        state.metadata["supported_information"] = list(report.supported_information)
        if restored_ranks:
            state.metadata["multi_document_coverage"] = {
                "restored_evidence_ranks": list(restored_ranks),
                "policy": "prior_need_retained_distinct_in_scope_documents",
            }
        return state


def preserve_multi_document_coverage(
    state: QueryState,
    *,
    prior: EvidenceGradingReport,
    final: EvidenceGradingReport,
) -> tuple[EvidenceGradingReport, tuple[int, ...]]:
    """Retain one need-level-approved safe item per scoped document in coverage mode."""

    if (
        state.coverage_mode is not CoverageMode.MULTI_DOCUMENT
        or not state.document_scope.strict
        or len(state.document_scope.allowed_document_ids) < 2
        or state.top_k < 2
    ):
        return final, ()

    evidence_by_rank = {item.rank: item for item in state.retrieved_evidence}
    prior_by_rank = {grade.evidence_rank: grade for grade in prior.grades}
    final_by_rank = {grade.evidence_rank: grade for grade in final.grades}
    final_documents = {
        item.document_id
        for rank, grade in final_by_rank.items()
        if grade.relevant
        and (item := evidence_by_rank.get(rank)) is not None
        and item.text.strip()
        and _scope_safe(state, item)
    }
    candidates_by_document: dict[object, list[EvidenceGrade]] = {}
    for rank, grade in sorted(prior_by_rank.items()):
        item = evidence_by_rank.get(rank)
        if (
            not grade.relevant
            or not grade.supports_information_need_ids
            or item is None
            or not item.text.strip()
            or not _scope_safe(state, item)
            or item.document_id is None
        ):
            continue
        candidates_by_document.setdefault(item.document_id, []).append(grade)

    restored: list[int] = []
    maximum_documents = min(
        state.top_k,
        len(state.document_scope.allowed_document_ids),
    )
    for document_id, grades in candidates_by_document.items():
        if document_id in final_documents or len(final_documents) >= maximum_documents:
            continue
        selected = min(grades, key=lambda grade: grade.evidence_rank)
        final_grade = final_by_rank.get(selected.evidence_rank)
        if final_grade is None or final_grade.relevant:
            continue
        final_by_rank[selected.evidence_rank] = EvidenceGrade(
            evidence_rank=selected.evidence_rank,
            relevance_score=selected.relevance_score,
            relevant=True,
            rationale=(
                "Retained for multi-document coverage because prior need-level grading "
                "retained this distinct in-scope document."
            ),
            supports_information_need_ids=selected.supports_information_need_ids,
        )
        restored.append(selected.evidence_rank)
        final_documents.add(document_id)

    if not restored:
        return final, ()

    prior_needs = {item.information_need_id: item for item in prior.information_need_grades}
    information_need_grades: list[InformationNeedGrade] = []
    for need in final.information_need_grades:
        added = tuple(
            rank
            for rank in restored
            if need.information_need_id
            in final_by_rank[rank].supports_information_need_ids
        )
        supporting = tuple(dict.fromkeys((*need.supporting_evidence_ranks, *added)))
        prior_need = prior_needs.get(need.information_need_id)
        status = need.status
        coverage_score = need.coverage_score
        if status is InformationNeedSupport.MISSING and supporting and prior_need is not None:
            status = prior_need.status
            coverage_score = prior_need.coverage_score
        information_need_grades.append(
            InformationNeedGrade(
                information_need_id=need.information_need_id,
                description=need.description,
                status=status,
                coverage_score=coverage_score,
                supporting_evidence_ranks=supporting,
                rationale=need.rationale,
                required=need.required,
            ),
        )

    required = tuple(item for item in information_need_grades if item.required)
    relevant_count = sum(grade.relevant for grade in final_by_rank.values())
    status = (
        EvidenceSufficiency.MISSING
        if relevant_count == 0
        else EvidenceSufficiency.SUFFICIENT
        if all(item.supported for item in required)
        else EvidenceSufficiency.WEAK
    )
    coverage_score = (
        sum(item.coverage_score for item in required) / len(required)
        if required
        else 1.0
    )
    report = EvidenceGradingReport(
        status=status,
        coverage_score=coverage_score,
        grades=tuple(final_by_rank[rank] for rank in sorted(final_by_rank)),
        information_need_grades=tuple(information_need_grades),
        rationale=(
            f"{final.rationale} Multi-document coverage retained prior need-level "
            f"ranks {sorted(restored)} from distinct in-scope documents."
        ),
        grader_name=final.grader_name,
        fallback_used=final.fallback_used,
        structured_output=final.structured_output,
    )
    return report, tuple(sorted(restored))


def _scope_safe(state: QueryState, item: EvidenceItem) -> bool:
    if not state.document_scope.allows(item.document_id):
        return False
    if not state.subject_lanes:
        return True
    lane = next(
        (
            candidate
            for candidate in state.subject_lanes
            if candidate.lane_id == item.subject_lane_id
        ),
        None,
    )
    return bool(
        lane is not None
        and item.subject_id == lane.subject_id
        and item.subject_name == lane.subject_name
        and lane.document_scope.allows(item.document_id)
    )


def _information_needs(state: QueryState) -> tuple[InformationNeed, ...]:
    decomposition = state.information_need_decomposition
    if decomposition is not None and decomposition.information_needs:
        return decomposition.information_needs
    return (
        InformationNeed(
            need_id="need_1",
            description=state.question,
            retrieval_query=state.question,
        ),
    )


def _query_constraints(state: QueryState) -> RetrievalConstraints:
    classification = state.query_classification
    if classification is not None:
        return RetrievalConstraints(
            document=classification.document_constraint,
            version=classification.version_constraint,
            dates=classification.date_constraints,
            document_scope=state.document_scope,
        )
    return RetrievalConstraints(document_scope=state.document_scope)
