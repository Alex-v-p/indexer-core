from __future__ import annotations

import time
from datetime import UTC, datetime

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.evaluation.models import EvaluationCase, EvaluationDataset
from packages.rag_core.evaluation.stability import (
    StabilityAttemptSnapshot,
    StabilityCaseResult,
    StabilityEvaluationReport,
    calculate_stability_metrics,
    canonical_route_signature,
    outcome_for_state,
    safe_runtime_profile,
    safe_structured_diagnostics,
    stable_evidence_identity,
)
from packages.rag_core.generation import CitationItem
from packages.rag_core.pipelines import RetrievalPipeline
from packages.rag_core.retrieval.models import EvidenceItem


class StabilityEvaluationRunner:
    """Run each dataset case repeatedly without changing the one-shot evaluator."""

    def __init__(
        self,
        *,
        graph: RetrievalPipeline,
        requested_pipeline_name: str | None = None,
    ) -> None:
        self._graph = graph
        self._requested_pipeline_name = requested_pipeline_name

    async def run(
        self,
        dataset: EvaluationDataset,
        *,
        repetitions: int,
        top_k_override: int | None = None,
    ) -> StabilityEvaluationReport:
        if repetitions < 2:
            raise ValueError("Stability evaluation requires repetitions >= 2.")
        if top_k_override is not None and top_k_override <= 0:
            raise ValueError("top_k_override must be positive.")

        started_at = datetime.now(UTC)
        started = time.perf_counter()
        cases: list[StabilityCaseResult] = []
        for case in dataset.cases:
            top_k = top_k_override or case.top_k or dataset.default_top_k
            attempt_results: list[StabilityAttemptSnapshot] = []
            for attempt_number in range(1, repetitions + 1):
                attempt_results.append(
                    await self._run_attempt(
                        case,
                        attempt_number=attempt_number,
                        top_k=top_k,
                    ),
                )
            attempts = tuple(attempt_results)
            cases.append(
                StabilityCaseResult(
                    case_id=case.id,
                    question=case.question,
                    top_k=top_k,
                    repetitions=repetitions,
                    attempts=attempts,
                    metrics=calculate_stability_metrics((attempts,)),
                    tags=case.tags,
                    metadata=case.metadata,
                ),
            )

        completed_at = datetime.now(UTC)
        duration_ms = int((time.perf_counter() - started) * 1000)
        all_attempts = [attempt for case in cases for attempt in case.attempts]
        succeeded = sum(attempt.status == "succeeded" for attempt in all_attempts)
        runtime_profile = {
            "pipeline_name": self._graph.name,
            "pipeline_version": self._graph.version,
        }
        for attempt in all_attempts:
            if attempt.runtime_profile:
                runtime_profile.update(
                    {
                        key: value
                        for key, value in attempt.runtime_profile.items()
                        if value is not None
                    },
                )
                break
        return StabilityEvaluationReport(
            schema_version="1.0",
            report_type="repeated_query_stability",
            dataset_schema_version=dataset.schema_version,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_description=dataset.description,
            dataset_metadata=dataset.metadata,
            pipeline_name=self._graph.name,
            pipeline_version=self._graph.version,
            repetitions=repetitions,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_ms=duration_ms,
            total_cases=len(cases),
            total_attempts=len(all_attempts),
            succeeded_attempts=succeeded,
            failed_attempts=len(all_attempts) - succeeded,
            runtime_profile=runtime_profile,
            metrics=calculate_stability_metrics([case.attempts for case in cases]),
            cases=tuple(cases),
        )

    async def _run_attempt(
        self,
        case: EvaluationCase,
        *,
        attempt_number: int,
        top_k: int,
    ) -> StabilityAttemptSnapshot:
        started = time.perf_counter()
        state = QueryState(
            question=case.question,
            top_k=top_k,
            requested_pipeline_name=self._requested_pipeline_name,
        )
        try:
            state = await self._graph.run(state)
            status = "succeeded"
            error_type = None
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__

        duration_ms = int((time.perf_counter() - started) * 1000)
        presentation = (
            state.answer_presentation.to_metadata()
            if state.answer_presentation is not None
            else None
        )
        return StabilityAttemptSnapshot(
            attempt_number=attempt_number,
            status=status,
            top_k=top_k,
            answer=state.answer,
            answer_presentation=presentation,
            outcome=outcome_for_state(state) if status == "succeeded" else None,
            route_signature=canonical_route_signature(state),
            evidence_identities=tuple(
                sorted({stable_evidence_identity(item) for item in state.retrieved_evidence}),
            ),
            evidence=tuple(_evidence_snapshot(item) for item in state.retrieved_evidence),
            citations=tuple(_citation_snapshot(item) for item in state.citations),
            structured_diagnostics=safe_structured_diagnostics(_diagnostic_metadata(state)),
            runtime_profile=safe_runtime_profile(state),
            duration_ms=duration_ms,
            error_type=error_type,
        )


def _diagnostic_metadata(state: QueryState) -> dict[str, object]:
    metadata: dict[str, object] = dict(state.metadata)
    typed_reports = (
        ("query_classification", state.query_classification),
        ("information_need_decomposition", state.information_need_decomposition),
        ("evidence_grading", state.evidence_grading),
        ("retrieval_retry", state.retrieval_retry),
        ("information_need_resolution", state.information_need_resolution),
    )
    for key, value in typed_reports:
        if value is not None and key not in metadata:
            metadata[key] = value.to_metadata()
    return metadata


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


def _string_or_none(value: object) -> str | None:
    return None if value is None else str(value)
