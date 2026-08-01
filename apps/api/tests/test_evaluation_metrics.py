from __future__ import annotations

import uuid

from packages.rag_core.generation import CitationItem
from packages.rag_core.evaluation.metrics import PlaceholderFaithfulnessEvaluator, calculate_case_metrics
from packages.rag_core.document_scope import DocumentScope, SubjectDocumentLane
from packages.rag_core.evaluation.models import BehavioralExpectations, EvidenceExpectation
from packages.rag_core.retrieval import EvidenceItem


async def test_metrics_calculate_recall_mrr_and_citation_hit_rate() -> None:
    expectations = (
        EvidenceExpectation(text_contains=("relevant alpha",)),
        EvidenceExpectation(metadata={"section_title": "Beta"}),
    )
    evidence = [
        EvidenceItem(rank=1, text="Unrelated chunk", metadata={"section_title": "Other"}),
        EvidenceItem(rank=2, text="This is relevant alpha evidence."),
        EvidenceItem(rank=3, text="Second relevant chunk", metadata={"section_title": "Beta"}),
    ]
    citations = [
        CitationItem(citation_index=1, evidence_rank=2),
        CitationItem(citation_index=2, evidence_rank=1),
    ]
    faithfulness = await PlaceholderFaithfulnessEvaluator().evaluate(
        question="Question",
        answer="Answer",
        evidence=evidence,
    )

    metrics = calculate_case_metrics(
        expectations=expectations,
        evidence=evidence,
        citations=citations,
        top_k=2,
        faithfulness=faithfulness,
    )

    assert metrics.recall_at_k.value == 0.5
    assert metrics.reciprocal_rank.value == 0.5
    assert metrics.citation_hit_rate.value == 0.5
    assert metrics.answer_faithfulness.status == "not_implemented"


async def test_metrics_are_not_applicable_without_expected_evidence() -> None:
    faithfulness = await PlaceholderFaithfulnessEvaluator().evaluate(question="Question", answer="Answer", evidence=[])

    metrics = calculate_case_metrics(
        expectations=(),
        evidence=[],
        citations=[],
        top_k=5,
        faithfulness=faithfulness,
    )

    assert metrics.recall_at_k.status == "not_applicable"
    assert metrics.reciprocal_rank.status == "not_applicable"
    assert metrics.citation_hit_rate.status == "not_applicable"
    assert metrics.answer_faithfulness.status == "not_implemented"


async def test_subject_scope_metrics_are_deterministic() -> None:
    daf_subject = uuid.UUID("10000000-0000-0000-0000-000000000001")
    internship_subject = uuid.UUID("10000000-0000-0000-0000-000000000002")
    daf_document = uuid.UUID("20000000-0000-0000-0000-000000000001")
    internship_document = uuid.UUID("20000000-0000-0000-0000-000000000003")
    outside_document = uuid.UUID("20000000-0000-0000-0000-000000000004")
    lanes = (
        SubjectDocumentLane(daf_subject, "DAF", DocumentScope.strict_scope((daf_document,))),
        SubjectDocumentLane(
            internship_subject,
            "Large Internship",
            DocumentScope.strict_scope((internship_document,)),
        ),
    )
    evidence = [
        EvidenceItem(rank=1, text="DAF", document_id=daf_document, subject_lane_id=lanes[0].lane_id, subject_id=daf_subject, subject_name="DAF"),
        EvidenceItem(rank=2, text="Internship", document_id=internship_document, subject_lane_id=lanes[1].lane_id, subject_id=internship_subject, subject_name="Large Internship"),
        EvidenceItem(rank=3, text="Outside", document_id=outside_document, metadata={"original_filename": "outside-scope.md"}),
    ]
    citations = [
        CitationItem(citation_index=1, evidence_rank=1, document_id=daf_document, subject_lane_id=lanes[0].lane_id, subject_id=daf_subject, subject_name="DAF"),
        CitationItem(citation_index=2, evidence_rank=3, document_id=outside_document, subject_lane_id=lanes[0].lane_id),
    ]
    faithfulness = await PlaceholderFaithfulnessEvaluator().evaluate(
        question="Compare", answer="Answer", evidence=evidence,
    )
    behavior = BehavioralExpectations(
        expected_product_outcome="answered",
        expected_scope_subject_names=("DAF", "Large Internship"),
        forbidden_document_ids=(str(outside_document),),
        max_scope_leakage=0,
        minimum_distinct_relevant_documents=2,
        expected_lane_subject_names=("DAF", "Large Internship"),
        minimum_evidence_per_lane=1,
        require_citation_scope_validity=True,
    )
    metrics = calculate_case_metrics(
        expectations=(EvidenceExpectation(document_id=str(daf_document)), EvidenceExpectation(document_id=str(internship_document))),
        evidence=evidence,
        citations=citations,
        top_k=3,
        faithfulness=faithfulness,
        behavioral_expectations=behavior,
        actual_product_outcome="answered",
        actual_subject_scope={
            "matched_subject_ids": [str(daf_subject), str(internship_subject)],
            "catalog": [
                {"subject_id": str(daf_subject), "name": "DAF"},
                {"subject_id": str(internship_subject), "name": "Large Internship"},
            ],
            "document_scope": {"global": False},
        },
        document_scope=DocumentScope.strict_scope((daf_document, internship_document)),
        subject_lanes=lanes,
    )

    assert metrics.scope_leakage_count.value == 1.0
    assert metrics.scope_leakage_rate.value == 1 / 3
    assert metrics.clarification_correctness.value == 1.0
    assert metrics.subject_scope_accuracy.value == 1.0
    assert metrics.lane_coverage.value == 1.0
    assert metrics.document_diversity.value == 2.0
    assert metrics.citation_scope_violations.value == 1.0
    assert metrics.citation_scope_validity.value == 0.0


async def test_lane_and_citation_provenance_reject_contradictory_attribution() -> None:
    daf_subject = uuid.UUID("10000000-0000-0000-0000-000000000001")
    internship_subject = uuid.UUID("10000000-0000-0000-0000-000000000002")
    daf_document = uuid.UUID("20000000-0000-0000-0000-000000000001")
    internship_document = uuid.UUID("20000000-0000-0000-0000-000000000002")
    lanes = (
        SubjectDocumentLane(daf_subject, "DAF", DocumentScope.strict_scope((daf_document,))),
        SubjectDocumentLane(internship_subject, "Large Internship", DocumentScope.strict_scope((internship_document,))),
    )
    contradictory = EvidenceItem(
        rank=1,
        text="contradictory",
        document_id=internship_document,
        subject_lane_id=lanes[0].lane_id,
        subject_id=internship_subject,
        subject_name="Large Internship",
    )
    citation = CitationItem(
        citation_index=1,
        evidence_rank=1,
        document_id=internship_document,
        subject_lane_id=lanes[0].lane_id,
        subject_id=internship_subject,
        subject_name="Large Internship",
    )
    faithfulness = await PlaceholderFaithfulnessEvaluator().evaluate(
        question="Compare", answer="Answer", evidence=[contradictory],
    )

    metrics = calculate_case_metrics(
        expectations=(), evidence=[contradictory], citations=[citation], top_k=1,
        faithfulness=faithfulness,
        behavioral_expectations=BehavioralExpectations(
            expected_lane_subject_names=("DAF",),
            require_citation_scope_validity=True,
            max_scope_leakage=0,
        ),
        document_scope=DocumentScope.strict_scope((daf_document, internship_document)),
        subject_lanes=lanes,
    )

    assert metrics.scope_leakage_count.value == 1.0
    assert metrics.lane_coverage.value == 0.0
    assert metrics.citation_scope_violations.value == 1.0
    assert metrics.citation_scope_validity.value == 0.0
