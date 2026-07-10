from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from packages.rag_core.agents.state import CitationItem, QueryState, TraceEvent
from packages.rag_core.evaluation.metrics import (
    PlaceholderFaithfulnessEvaluator,
    aggregate_case_metrics,
    calculate_case_metrics,
    failed_case_metrics,
)
from packages.rag_core.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationReport,
    MetricValue,
)
from packages.rag_core.pipelines import RetrievalPipeline
from packages.rag_core.retrieval.models import EvidenceItem


class FaithfulnessEvaluator(Protocol):
    async def evaluate(self, *, question: str, answer: str | None, evidence: Sequence[EvidenceItem]) -> MetricValue:
        """Score whether an answer is supported by the retrieved evidence."""


class EvaluationRunner:
    """Run a dataset through the same complete graph boundary used by the API."""

    def __init__(
        self,
        *,
        graph: RetrievalPipeline,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
        requested_pipeline_name: str | None = None,
    ) -> None:
        self._graph = graph
        self._faithfulness_evaluator = faithfulness_evaluator or PlaceholderFaithfulnessEvaluator()
        self._requested_pipeline_name = requested_pipeline_name

    async def run(self, dataset: EvaluationDataset, *, top_k_override: int | None = None) -> EvaluationReport:
        if top_k_override is not None and top_k_override <= 0:
            raise ValueError("top_k_override must be positive.")

        started_at = datetime.now(UTC)
        started = time.perf_counter()
        results: list[EvaluationCaseResult] = []
        for case in dataset.cases:
            top_k = top_k_override or case.top_k or dataset.default_top_k
            results.append(await self._run_case(case, top_k=top_k))

        completed_at = datetime.now(UTC)
        duration_ms = int((time.perf_counter() - started) * 1000)
        succeeded_cases = sum(result.status == "succeeded" for result in results)
        return EvaluationReport(
            schema_version="1.0",
            dataset_schema_version=dataset.schema_version,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_description=dataset.description,
            dataset_metadata=dataset.metadata,
            pipeline_name=self._graph.name,
            pipeline_version=self._graph.version,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_ms=duration_ms,
            total_cases=len(results),
            succeeded_cases=succeeded_cases,
            failed_cases=len(results) - succeeded_cases,
            metrics=aggregate_case_metrics([result.metrics for result in results]),
            cases=tuple(results),
        )

    async def _run_case(self, case: EvaluationCase, *, top_k: int) -> EvaluationCaseResult:
        started = time.perf_counter()
        state = QueryState(
            question=case.question,
            top_k=top_k,
            requested_pipeline_name=self._requested_pipeline_name,
        )
        try:
            state = await self._graph.run(state)
            faithfulness = await self._faithfulness_evaluator.evaluate(
                question=case.question,
                answer=state.answer,
                evidence=state.retrieved_evidence,
            )
            metrics = calculate_case_metrics(
                expectations=case.expected_evidence,
                evidence=state.retrieved_evidence,
                citations=state.citations,
                top_k=top_k,
                faithfulness=faithfulness,
            )
            status = "succeeded"
            error_message = None
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            metrics = failed_case_metrics(error_message)

        duration_ms = int((time.perf_counter() - started) * 1000)
        return EvaluationCaseResult(
            case_id=case.id,
            question=case.question,
            top_k=top_k,
            status=status,
            duration_ms=duration_ms,
            expected_answer=case.expected_answer,
            expected_evidence=case.expected_evidence,
            actual_answer=state.answer,
            evidence=tuple(_evidence_snapshot(item) for item in state.retrieved_evidence),
            citations=tuple(_citation_snapshot(item) for item in state.citations),
            trace=tuple(_trace_snapshot(item) for item in state.trace),
            metrics=metrics,
            tags=case.tags,
            metadata=case.metadata,
            error_message=error_message,
        )


def _evidence_snapshot(evidence: EvidenceItem) -> dict[str, object]:
    return {
        "rank": evidence.rank,
        "score": evidence.score,
        "text": evidence.text,
        "qdrant_chunk_index_id": _string_or_none(evidence.qdrant_chunk_index_id),
        "document_id": _string_or_none(evidence.document_id),
        "document_version_id": _string_or_none(evidence.document_version_id),
        "metadata": evidence.metadata,
    }


def _citation_snapshot(citation: CitationItem) -> dict[str, object]:
    return {
        "citation_index": citation.citation_index,
        "evidence_rank": citation.evidence_rank,
        "label": citation.label,
        "page_number": citation.page_number,
        "quote": citation.quote,
        "qdrant_chunk_index_id": _string_or_none(citation.qdrant_chunk_index_id),
        "document_id": _string_or_none(citation.document_id),
        "document_version_id": _string_or_none(citation.document_version_id),
        "metadata": citation.metadata,
    }


def _trace_snapshot(trace: TraceEvent) -> dict[str, object]:
    return {
        "step_order": trace.step_order,
        "name": trace.name,
        "step_type": trace.step_type,
        "status": trace.status,
        "duration_ms": trace.duration_ms,
        "input_summary": trace.input_summary,
        "output_summary": trace.output_summary,
        "error_message": trace.error_message,
        "metadata": trace.metadata,
    }


def _string_or_none(value: object) -> str | None:
    return None if value is None else str(value)
