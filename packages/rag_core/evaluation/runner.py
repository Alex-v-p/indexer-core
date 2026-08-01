from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.document_scope import (
    CoverageMode,
    QueryProductOutcome,
    ResolvedSubjectScopeSnapshot,
)
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
from packages.rag_core.evaluation.snapshots import (
    citation_snapshot,
    evidence_snapshot,
    trace_snapshot,
)


class FaithfulnessEvaluator(Protocol):
    async def evaluate(self, *, question: str, answer: str | None, evidence: Sequence[EvidenceItem]) -> MetricValue:
        """Score whether an answer is supported by the retrieved evidence."""


@dataclass(frozen=True, slots=True)
class SubjectScopeEvaluationRequest:
    question: str
    requested_subject_ids: tuple[uuid.UUID, ...]
    requested_subject_names: tuple[str, ...]
    coverage_mode: CoverageMode


class SubjectScopeEvaluationProvider(Protocol):
    """Application-owned adapter for production-equivalent subject resolution."""

    async def resolve(
        self,
        request: SubjectScopeEvaluationRequest,
    ) -> ResolvedSubjectScopeSnapshot: ...


class EvaluationRunner:
    """Run a dataset through the same complete graph boundary used by the API."""

    def __init__(
        self,
        *,
        graph: RetrievalPipeline,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
        requested_pipeline_name: str | None = None,
        subject_scope_provider: SubjectScopeEvaluationProvider | None = None,
    ) -> None:
        self._graph = graph
        self._faithfulness_evaluator = faithfulness_evaluator or PlaceholderFaithfulnessEvaluator()
        self._requested_pipeline_name = requested_pipeline_name
        self._subject_scope_provider = subject_scope_provider

    async def run(self, dataset: EvaluationDataset, *, top_k_override: int | None = None) -> EvaluationReport:
        if top_k_override is not None and top_k_override <= 0:
            raise ValueError("top_k_override must be positive.")
        dataset_scope_opt_in = _dataset_scope_opt_in(dataset)
        scoped = evaluation_dataset_requires_subject_scope(dataset)
        if scoped:
            if self._subject_scope_provider is None:
                raise ValueError(
                    "Subject-scoping evaluation requires a SubjectScopeEvaluationProvider; "
                    "baseline datasets without subject expectations remain global.",
                )
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        results: list[EvaluationCaseResult] = []
        for case in dataset.cases:
            top_k = top_k_override or case.top_k or dataset.default_top_k
            results.append(
                await self._run_case(
                    case,
                    top_k=top_k,
                    dataset_scope_opt_in=dataset_scope_opt_in,
                ),
            )

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

    async def _run_case(
        self,
        case: EvaluationCase,
        *,
        top_k: int,
        dataset_scope_opt_in: bool,
    ) -> EvaluationCaseResult:
        started = time.perf_counter()
        state = QueryState(
            question=case.question,
            top_k=top_k,
            requested_pipeline_name=self._requested_pipeline_name,
            coverage_mode=CoverageMode(case.coverage_mode),
            metadata={
                "evaluation_request": {
                    "requested_subject_ids": list(case.requested_subject_ids),
                    "requested_subject_names": list(case.requested_subject_names),
                    "coverage_mode": case.coverage_mode,
                }
            },
        )
        try:
            if _requires_subject_scope(
                case,
                dataset_scope_opt_in=dataset_scope_opt_in,
            ):
                if self._subject_scope_provider is None:  # guarded by run preflight
                    raise RuntimeError("Subject scope provider is unavailable.")
                snapshot = await self._subject_scope_provider.resolve(
                    SubjectScopeEvaluationRequest(
                        question=case.question,
                        requested_subject_ids=tuple(uuid.UUID(item) for item in case.requested_subject_ids),
                        requested_subject_names=case.requested_subject_names,
                        coverage_mode=CoverageMode(case.coverage_mode),
                    ),
                )
                state.document_scope = snapshot.document_scope
                state.subject_lanes = snapshot.subject_lanes
                state.coverage_mode = snapshot.coverage_mode
                state.comparison_requested = snapshot.comparison_requested
                state.product_outcome = snapshot.product_outcome
                state.metadata["resolved_subject_scope"] = snapshot.to_metadata()
                if snapshot.product_outcome is not None:
                    state.metadata["query_product_outcome"] = snapshot.product_outcome.value
            if state.product_outcome not in {
                QueryProductOutcome.CLARIFICATION_REQUIRED,
                QueryProductOutcome.NO_EVIDENCE,
            }:
                state = await self._graph.run(state)
                if state.product_outcome is None:
                    state.product_outcome = (
                        QueryProductOutcome.ANSWERED
                        if state.retrieved_evidence
                        else QueryProductOutcome.NO_EVIDENCE
                    )
                    state.metadata["query_product_outcome"] = state.product_outcome.value
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
                behavioral_expectations=case.behavioral_expectations,
                actual_product_outcome=_product_outcome(state),
                actual_subject_scope=_subject_scope(state),
                document_scope=state.document_scope,
                subject_lanes=state.subject_lanes,
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
            behavioral_expectations=case.behavioral_expectations,
            actual_answer=state.answer,
            actual_product_outcome=_product_outcome(state),
            actual_subject_scope=_subject_scope(state),
            evidence=tuple(evidence_snapshot(item) for item in state.retrieved_evidence),
            citations=tuple(citation_snapshot(item) for item in state.citations),
            trace=tuple(trace_snapshot(item) for item in state.trace),
            metrics=metrics,
            tags=case.tags,
            metadata=case.metadata,
            error_message=error_message,
        )


def _product_outcome(state: QueryState) -> str | None:
    if state.product_outcome is not None:
        return state.product_outcome.value
    value = state.metadata.get("query_product_outcome")
    return value if isinstance(value, str) else None


def _subject_scope(state: QueryState) -> dict[str, object] | None:
    value = state.metadata.get("resolved_subject_scope")
    return dict(value) if isinstance(value, dict) else None


def _requires_subject_scope(
    case: EvaluationCase,
    *,
    dataset_scope_opt_in: bool = False,
) -> bool:
    behavior = case.behavioral_expectations
    return bool(
        dataset_scope_opt_in
        or case.metadata.get("subject_scope_evaluation") is True
        or case.requested_subject_ids
        or case.requested_subject_names
        or behavior.expected_scope_subject_ids
        or behavior.expected_scope_subject_names
        or behavior.expect_global_scope is not None
        or behavior.expected_lane_subject_ids
        or behavior.expected_lane_subject_names
    )


def evaluation_dataset_requires_subject_scope(dataset: EvaluationDataset) -> bool:
    dataset_scope_opt_in = _dataset_scope_opt_in(dataset)
    return any(
        _requires_subject_scope(
            case,
            dataset_scope_opt_in=dataset_scope_opt_in,
        )
        for case in dataset.cases
    )


def _dataset_scope_opt_in(dataset: EvaluationDataset) -> bool:
    return dataset.metadata.get("subject_scope_evaluation") is True

