from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from packages.rag_core.agents.state import CitationItem
from packages.rag_core.evaluation.matching import evidence_matches, first_relevant_rank, maximum_expectation_matches
from packages.rag_core.evaluation.models import AggregateMetrics, CaseMetrics, EvidenceExpectation, MetricValue
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
) -> CaseMetrics:
    if not expectations:
        not_applicable = MetricValue(
            value=None,
            status="not_applicable",
            details="This case has no expected evidence annotations.",
        )
        return CaseMetrics(
            recall_at_k=not_applicable,
            reciprocal_rank=not_applicable,
            citation_hit_rate=not_applicable,
            answer_faithfulness=faithfulness,
        )

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

    return CaseMetrics(
        recall_at_k=MetricValue(
            value=recall,
            status="computed",
            details=f"Matched {len(matched)} of {len(expectations)} expected evidence items within rank {top_k}.",
        ),
        reciprocal_rank=MetricValue(
            value=reciprocal_rank,
            status="computed",
            details="Reciprocal rank of the first retrieved item matching any expected evidence annotation.",
        ),
        citation_hit_rate=MetricValue(
            value=citation_hit_rate,
            status="computed",
            details=f"{citation_hits} of {len(citations)} emitted citations linked to expected evidence.",
        ),
        answer_faithfulness=faithfulness,
    )


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
    )


def failed_case_metrics(error_message: str) -> CaseMetrics:
    failed = MetricValue(value=None, status="failed", details=error_message)
    return CaseMetrics(
        recall_at_k=failed,
        reciprocal_rank=failed,
        citation_hit_rate=failed,
        answer_faithfulness=failed,
    )


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
