from __future__ import annotations

from packages.rag_core.generation import CitationItem
from packages.rag_core.evaluation.metrics import PlaceholderFaithfulnessEvaluator, calculate_case_metrics
from packages.rag_core.evaluation.models import EvidenceExpectation
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
