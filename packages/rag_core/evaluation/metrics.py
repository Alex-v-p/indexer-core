from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from packages.rag_core.generation import CitationItem
from packages.rag_core.document_scope import DocumentScope, SubjectDocumentLane
from packages.rag_core.evaluation.matching import (
    evidence_matches,
    first_relevant_rank,
    matched_scope_subjects,
    maximum_expectation_matches,
    source_is_forbidden,
)
from packages.rag_core.evaluation.models import (
    AggregateMetrics,
    BehavioralExpectations,
    CaseMetrics,
    EvidenceExpectation,
    MetricValue,
)
from packages.rag_core.retrieval.models import EvidenceItem


class PlaceholderFaithfulnessEvaluator:
    """Explicit extension point for a future groundedness/faithfulness judge."""

    async def evaluate(self, *, question: str, answer: str | None, evidence: Sequence[EvidenceItem]) -> MetricValue:
        del question, answer, evidence
        return MetricValue(
            value=None,
            status="not_implemented",
            details="Answer faithfulness scoring is reserved for a later evaluator implementation.",
        )


def calculate_case_metrics(
    *,
    expectations: Sequence[EvidenceExpectation],
    evidence: Sequence[EvidenceItem],
    citations: Sequence[CitationItem],
    top_k: int,
    faithfulness: MetricValue,
    behavioral_expectations: BehavioralExpectations = BehavioralExpectations(),
    actual_product_outcome: str | None = None,
    actual_subject_scope: dict[str, object] | None = None,
    document_scope: DocumentScope = DocumentScope(),
    subject_lanes: Sequence[SubjectDocumentLane] = (),
) -> CaseMetrics:
    retrieval = _retrieval_metrics(
        expectations=expectations,
        evidence=evidence,
        citations=citations,
        top_k=top_k,
    )
    behavior = _behavior_metrics(
        expectations=behavioral_expectations,
        evidence_expectations=expectations,
        evidence=evidence,
        citations=citations,
        actual_product_outcome=actual_product_outcome,
        actual_subject_scope=actual_subject_scope,
        document_scope=document_scope,
        subject_lanes=subject_lanes,
    )
    return CaseMetrics(
        recall_at_k=retrieval[0],
        reciprocal_rank=retrieval[1],
        citation_hit_rate=retrieval[2],
        answer_faithfulness=faithfulness,
        **behavior,
    )


def _retrieval_metrics(
    *,
    expectations: Sequence[EvidenceExpectation],
    evidence: Sequence[EvidenceItem],
    citations: Sequence[CitationItem],
    top_k: int,
) -> tuple[MetricValue, MetricValue, MetricValue]:
    if not expectations:
        not_applicable = MetricValue(
            value=None,
            status="not_applicable",
            details="This case has no expected evidence annotations.",
        )
        return not_applicable, not_applicable, not_applicable

    ranked_evidence = sorted(evidence, key=lambda item: item.rank)
    evidence_at_k = [item for item in ranked_evidence if item.rank <= top_k]
    matched = maximum_expectation_matches(expectations, evidence_at_k)
    recall = len(matched) / len(expectations)

    first_rank = first_relevant_rank(expectations, ranked_evidence)
    reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank

    evidence_by_rank = {item.rank: item for item in ranked_evidence}
    citation_hits = 0
    for citation in citations:
        linked_evidence = evidence_by_rank.get(citation.evidence_rank or -1)
        if linked_evidence is not None and any(
            evidence_matches(expectation, linked_evidence) for expectation in expectations
        ):
            citation_hits += 1
    citation_hit_rate = citation_hits / len(citations) if citations else 0.0

    return (
        MetricValue(
            value=recall,
            status="computed",
            details=f"Matched {len(matched)} of {len(expectations)} expected evidence items within rank {top_k}.",
        ),
        MetricValue(
            value=reciprocal_rank,
            status="computed",
            details="Reciprocal rank of the first retrieved item matching any expected evidence annotation.",
        ),
        MetricValue(
            value=citation_hit_rate,
            status="computed",
            details=f"{citation_hits} of {len(citations)} emitted citations linked to expected evidence.",
        ),
    )


def _behavior_metrics(
    *,
    expectations: BehavioralExpectations,
    evidence_expectations: Sequence[EvidenceExpectation],
    evidence: Sequence[EvidenceItem],
    citations: Sequence[CitationItem],
    actual_product_outcome: str | None,
    actual_subject_scope: dict[str, object] | None,
    document_scope: DocumentScope,
    subject_lanes: Sequence[SubjectDocumentLane],
) -> dict[str, MetricValue]:
    not_applicable = MetricValue(None, "not_applicable", "No deterministic expectation was configured.")
    leakage_applicable = bool(
        expectations.forbidden_document_ids
        or expectations.forbidden_document_names
        or expectations.max_scope_leakage is not None
        or document_scope.strict
    )
    leaked = [
        item
        for item in evidence
        if _outside_scope(item.document_id, document_scope)
        or _evidence_lane_violation(item, subject_lanes)
        or source_is_forbidden(
            document_id=item.document_id,
            metadata=item.metadata,
            forbidden_document_ids=expectations.forbidden_document_ids,
            forbidden_document_names=expectations.forbidden_document_names,
        )
    ]
    leakage_details = (
        f"Observed {len(leaked)} leaked evidence item(s) from {len(evidence)} retrieved item(s)."
        + (
            f" Expected at most {expectations.max_scope_leakage}."
            if expectations.max_scope_leakage is not None
            else ""
        )
    )
    leakage_count = MetricValue(float(len(leaked)), "computed", leakage_details) if leakage_applicable else not_applicable
    leakage_rate = (
        MetricValue(len(leaked) / len(evidence) if evidence else 0.0, "computed", leakage_details)
        if leakage_applicable
        else not_applicable
    )

    expected_outcome = expectations.expected_product_outcome
    clarification = (
        MetricValue(
            1.0 if actual_product_outcome == expected_outcome else 0.0,
            "computed",
            f"Expected product outcome {expected_outcome!r}; observed {actual_product_outcome!r}.",
        )
        if expected_outcome is not None
        else not_applicable
    )

    actual_subject_ids, actual_subject_names = matched_scope_subjects(actual_subject_scope)
    expected_subject_ids = set(expectations.expected_scope_subject_ids)
    expected_subject_names = set(expectations.expected_scope_subject_names)
    scope_applicable = bool(
        expected_subject_ids
        or expected_subject_names
        or expectations.expect_global_scope is not None
    )
    actual_global = (
        actual_subject_scope.get("document_scope", {}).get("global")
        if isinstance(actual_subject_scope, dict)
        and isinstance(actual_subject_scope.get("document_scope"), dict)
        else None
    )
    scope_matches = (
        (not expected_subject_ids or actual_subject_ids == expected_subject_ids)
        and (not expected_subject_names or actual_subject_names == expected_subject_names)
        and (
            expectations.expect_global_scope is None
            or actual_global is expectations.expect_global_scope
        )
    )
    scope_accuracy = (
        MetricValue(
            1.0 if scope_matches else 0.0,
            "computed",
            f"Expected scope ids/names {sorted(expected_subject_ids)}/{sorted(expected_subject_names)}; "
            f"observed {sorted(actual_subject_ids)}/{sorted(actual_subject_names)}.",
        )
        if scope_applicable
        else not_applicable
    )

    expected_lanes = [
        *(('id', item) for item in expectations.expected_lane_subject_ids),
        *(('name', item) for item in expectations.expected_lane_subject_names),
    ]
    lane_hits = sum(
        _lane_evidence_count(kind, value, evidence, subject_lanes)
        >= expectations.minimum_evidence_per_lane
        for kind, value in expected_lanes
    )
    lane_coverage = (
        MetricValue(
            lane_hits / len(expected_lanes),
            "computed",
            f"{lane_hits} of {len(expected_lanes)} expected lanes met the minimum of "
            f"{expectations.minimum_evidence_per_lane} evidence item(s).",
        )
        if expected_lanes
        else not_applicable
    )

    relevant = [
        item
        for item in evidence
        if not evidence_expectations
        or any(evidence_matches(expectation, item) for expectation in evidence_expectations)
    ]
    distinct_documents = {
        str(item.document_id) if item.document_id is not None else _source_name(item.metadata)
        for item in relevant
        if item.document_id is not None or _source_name(item.metadata) is not None
    }
    minimum_documents = expectations.minimum_distinct_relevant_documents
    diversity = (
        MetricValue(
            float(len(distinct_documents)),
            "computed",
            f"Observed {len(distinct_documents)} distinct relevant document(s); expected at least {minimum_documents}.",
        )
        if minimum_documents is not None
        else not_applicable
    )

    citation_applicable = expectations.require_citation_scope_validity
    violations = [
        citation
        for citation in citations
        if _citation_scope_violation(
            citation,
            evidence=evidence,
            document_scope=document_scope,
            subject_lanes=subject_lanes,
            expectations=expectations,
        )
    ]
    citation_details = f"Observed {len(violations)} scope violation(s) across {len(citations)} citation(s)."
    citation_violations = (
        MetricValue(float(len(violations)), "computed", citation_details)
        if citation_applicable
        else not_applicable
    )
    citation_validity = (
        MetricValue(1.0 if not violations else 0.0, "computed", citation_details)
        if citation_applicable
        else not_applicable
    )
    return {
        "scope_leakage_count": leakage_count,
        "scope_leakage_rate": leakage_rate,
        "clarification_correctness": clarification,
        "subject_scope_accuracy": scope_accuracy,
        "lane_coverage": lane_coverage,
        "document_diversity": diversity,
        "citation_scope_violations": citation_violations,
        "citation_scope_validity": citation_validity,
    }


def aggregate_case_metrics(case_metrics: Sequence[CaseMetrics]) -> AggregateMetrics:
    return AggregateMetrics(
        recall_at_k=_mean_metric([metrics.recall_at_k for metrics in case_metrics], "Mean per-case recall@k."),
        mrr=_mean_metric([metrics.reciprocal_rank for metrics in case_metrics], "Mean reciprocal rank."),
        citation_hit_rate=_mean_metric(
            [metrics.citation_hit_rate for metrics in case_metrics],
            "Mean per-case citation hit rate.",
        ),
        answer_faithfulness=_mean_metric(
            [metrics.answer_faithfulness for metrics in case_metrics],
            "Mean answer faithfulness.",
        ),
        scope_leakage_count=_mean_metric([item.scope_leakage_count for item in case_metrics], "Mean scope leakage count."),
        scope_leakage_rate=_mean_metric([item.scope_leakage_rate for item in case_metrics], "Mean scope leakage rate."),
        clarification_correctness=_mean_metric([item.clarification_correctness for item in case_metrics], "Mean product-outcome correctness."),
        subject_scope_accuracy=_mean_metric([item.subject_scope_accuracy for item in case_metrics], "Mean resolved-subject-scope accuracy."),
        lane_coverage=_mean_metric([item.lane_coverage for item in case_metrics], "Mean comparison lane coverage."),
        document_diversity=_mean_metric([item.document_diversity for item in case_metrics], "Mean distinct relevant document count."),
        citation_scope_violations=_mean_metric([item.citation_scope_violations for item in case_metrics], "Mean citation scope violation count."),
        citation_scope_validity=_mean_metric([item.citation_scope_validity for item in case_metrics], "Mean citation scope validity."),
    )


def failed_case_metrics(error_message: str) -> CaseMetrics:
    failed = MetricValue(value=None, status="failed", details=error_message)
    return CaseMetrics(
        recall_at_k=failed,
        reciprocal_rank=failed,
        citation_hit_rate=failed,
        answer_faithfulness=failed,
        scope_leakage_count=failed,
        scope_leakage_rate=failed,
        clarification_correctness=failed,
        subject_scope_accuracy=failed,
        lane_coverage=failed,
        document_diversity=failed,
        citation_scope_violations=failed,
        citation_scope_validity=failed,
    )


def _outside_scope(document_id: object, scope: DocumentScope) -> bool:
    return scope.strict and (
        document_id is None or str(document_id) not in {str(item) for item in scope.allowed_document_ids}
    )


def _source_name(metadata: dict[str, object]) -> str | None:
    for key in ("original_filename", "document_name", "title"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _lane_evidence_count(
    kind: str,
    value: str,
    evidence: Sequence[EvidenceItem],
    subject_lanes: Sequence[SubjectDocumentLane],
) -> int:
    lane = next(
        (
            item
            for item in subject_lanes
            if (kind == "id" and str(item.subject_id) == value)
            or (kind == "name" and item.subject_name == value)
        ),
        None,
    )
    if lane is None:
        return 0
    return sum(not _evidence_lane_violation(item, (lane,)) for item in evidence)


def _evidence_lane_violation(
    evidence: EvidenceItem,
    subject_lanes: Sequence[SubjectDocumentLane],
) -> bool:
    if not subject_lanes:
        return False
    if evidence.subject_lane_id is None:
        return True
    lane = next((item for item in subject_lanes if item.lane_id == evidence.subject_lane_id), None)
    return bool(
        lane is None
        or evidence.subject_id != lane.subject_id
        or evidence.subject_name != lane.subject_name
        or _outside_scope(evidence.document_id, lane.document_scope)
    )


def _citation_scope_violation(
    citation: CitationItem,
    *,
    evidence: Sequence[EvidenceItem],
    document_scope: DocumentScope,
    subject_lanes: Sequence[SubjectDocumentLane],
    expectations: BehavioralExpectations,
) -> bool:
    linked = next(
        (item for item in evidence if citation.evidence_rank is not None and item.rank == citation.evidence_rank),
        None,
    )
    if linked is None:
        return True
    if (
        citation.document_id != linked.document_id
        or citation.subject_lane_id != linked.subject_lane_id
        or citation.subject_id != linked.subject_id
        or citation.subject_name != linked.subject_name
    ):
        return True
    if _evidence_lane_violation(linked, subject_lanes):
        return True
    if _outside_scope(citation.document_id, document_scope) or source_is_forbidden(
        document_id=citation.document_id,
        metadata=citation.metadata,
        forbidden_document_ids=expectations.forbidden_document_ids,
        forbidden_document_names=expectations.forbidden_document_names,
    ):
        return True
    if citation.subject_lane_id is None:
        return bool(subject_lanes)
    lane = next((item for item in subject_lanes if item.lane_id == citation.subject_lane_id), None)
    return lane is None or _outside_scope(citation.document_id, lane.document_scope)


def _mean_metric(values: Sequence[MetricValue], details: str) -> MetricValue:
    computed = [value.value for value in values if value.status == "computed" and value.value is not None]
    if computed:
        return MetricValue(
            value=fmean(computed),
            status="computed",
            details=f"{details} Computed from {len(computed)} applicable cases.",
        )
    if values and all(value.status == "not_implemented" for value in values):
        return MetricValue(value=None, status="not_implemented", details=values[0].details)
    return MetricValue(value=None, status="not_applicable", details=f"{details} No applicable successful cases.")
