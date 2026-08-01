from __future__ import annotations

import uuid

from packages.indexer_application.dto import QueryRunStatus
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.services.background_jobs.execution import ProgressReporter
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.runtime import GraphProgressEvent
from packages.rag_core.pipelines import PipelineRegistry, UnknownPipelineError
from packages.rag_core.document_scope import QueryProductOutcome
from packages.indexer_application.services.query_subject_scope import (
    QuerySubjectScopeConfig,
    resolve_or_reuse_query_subject_scope,
)


class ProcessQueryJobHandler:
    """Execute one complete query graph as a durable background job."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        pipeline_registry: PipelineRegistry,
        subject_scope_config: QuerySubjectScopeConfig = QuerySubjectScopeConfig(),
    ) -> None:
        self._uow = uow
        self._pipeline_registry = pipeline_registry
        self._subject_scope_config = subject_scope_config

    async def __call__(
        self,
        *,
        job_id: uuid.UUID,
        attempt: int,
        payload: dict[str, object],
        report: ProgressReporter,
    ) -> dict[str, object]:
        query_run_id = uuid.UUID(_required_string(payload, "query_run_id"))
        query_run = await self._uow.query_runs.get(query_run_id)
        if query_run is None:
            raise LookupError(f"Query run {query_run_id} was not found.")
        if query_run.status is QueryRunStatus.SUCCEEDED:
            await report(0.98, "query_result_already_available")
            return _result_payload(query_run_id, query_run.pipeline_name, query_run.status.value)

        await report(0.02, "resolving_subject_scope")
        query_run, subject_scope, snapshot_reused = await resolve_or_reuse_query_subject_scope(
            uow=self._uow,
            query_run_id=query_run_id,
            config=self._subject_scope_config,
        )
        # The immutable scope snapshot must be durable before any graph or
        # external retrieval call. Retries and stale workers only deserialize it.
        await self._uow.commit()

        pipeline_name = _required_string(payload, "pipeline_name")
        try:
            pipeline = self._pipeline_registry.build(pipeline_name)
        except UnknownPipelineError as exc:
            raise ValueError(str(exc)) from exc

        await report(0.04, "preparing_query_execution")
        await self._uow.query_runs.mark_running(
            query_run_id=query_run_id,
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
            background_job_id=job_id,
            attempt=attempt,
        )
        await self._uow.commit()

        if subject_scope.product_outcome in {
            QueryProductOutcome.CLARIFICATION_REQUIRED,
            QueryProductOutcome.NO_EVIDENCE,
        }:
            state = QueryState(
                question=query_run.question,
                top_k=query_run.top_k or _required_int(payload, "top_k"),
                query_run_id=query_run_id,
                requested_pipeline_name=_optional_string(
                    payload.get("requested_pipeline_name")
                ),
                pipeline_name=pipeline.name,
                pipeline_version=pipeline.version,
                document_scope=subject_scope.document_scope,
                subject_lanes=subject_scope.subject_lanes,
                coverage_mode=subject_scope.coverage_mode,
                comparison_requested=subject_scope.comparison_requested,
                product_outcome=subject_scope.product_outcome,
                metadata={
                    "resolved_subject_scope": subject_scope.to_metadata(),
                    "query_product_outcome": subject_scope.product_outcome.value,
                    "subject_scope_snapshot_reused": snapshot_reused,
                },
            )
            await self._uow.query_runs.mark_succeeded(
                query_run_id=query_run_id,
                state=state,
            )
            await self._uow.commit()
            await report(0.97, subject_scope.product_outcome.value)
            return _result_payload(
                query_run_id,
                pipeline.name,
                QueryRunStatus.SUCCEEDED.value,
                product_outcome=subject_scope.product_outcome,
            )

        tracker = QueryJobProgressTracker(report)
        state = QueryState(
            question=query_run.question,
            top_k=query_run.top_k or _required_int(payload, "top_k"),
            query_run_id=query_run_id,
            requested_pipeline_name=_optional_string(payload.get("requested_pipeline_name")),
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
            document_scope=subject_scope.document_scope,
            subject_lanes=subject_scope.subject_lanes,
            coverage_mode=subject_scope.coverage_mode,
            comparison_requested=subject_scope.comparison_requested,
            progress_observer=tracker,
            metadata={
                "resolved_subject_scope": subject_scope.to_metadata(),
                "subject_scope_snapshot_reused": snapshot_reused,
            },
        )
        state = await pipeline.run(state)
        state.product_outcome = (
            QueryProductOutcome.ANSWERED
            if state.retrieved_evidence
            else QueryProductOutcome.NO_EVIDENCE
        )
        state.metadata["query_product_outcome"] = state.product_outcome.value

        await report(0.97, "saving_query_result")
        await self._uow.query_runs.mark_succeeded(query_run_id=query_run_id, state=state)
        await self._uow.commit()
        return _result_payload(
            query_run_id,
            state.pipeline_name,
            QueryRunStatus.SUCCEEDED.value,
            product_outcome=state.product_outcome,
        )


class QueryJobProgressTracker:
    """Translate graph-node events into stable user-facing job stages."""

    def __init__(self, report: ProgressReporter) -> None:
        self._report = report
        self._progress = 0.04

    async def __call__(self, event: GraphProgressEvent) -> None:
        if event.status not in {"started", "failed"}:
            return
        progress, stage = _progress_for_event(event, current_progress=self._progress)
        self._progress = max(self._progress, progress)
        if event.status == "failed":
            stage = f"{stage}_failed"
        await self._report(self._progress, stage)


def _progress_for_event(
    event: GraphProgressEvent,
    *,
    current_progress: float,
) -> tuple[float, str]:
    if event.node_name in _INFORMATION_NEED_NODE_PROGRESS:
        total = _metadata_int(event.metadata, "information_need_count")
        completed = _metadata_int(event.metadata, "completed_information_need_count")
        if total is not None and total > 0 and completed is not None:
            local_fraction, stage = _INFORMATION_NEED_NODE_PROGRESS[event.node_name]
            bounded_completed = min(max(completed, 0), total - 1)
            slice_size = (_INFORMATION_NEED_PROGRESS_END - _INFORMATION_NEED_PROGRESS_START) / total
            return (
                _INFORMATION_NEED_PROGRESS_START
                + bounded_completed * slice_size
                + local_fraction * slice_size,
                stage,
            )
    return _NODE_PROGRESS.get(
        event.node_name,
        (max(current_progress, 0.30), "running_query_graph"),
    )


def _metadata_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


_INFORMATION_NEED_PROGRESS_START = 0.28
_INFORMATION_NEED_PROGRESS_END = 0.80

_INFORMATION_NEED_NODE_PROGRESS: dict[str, tuple[float, str]] = {
    "select_information_need": (0.00, "selecting_information_need"),
    "classify_information_need": (0.10, "classifying_information_need"),
    "plan_information_need": (0.24, "planning_retrieval"),
    "retrieve": (0.42, "retrieving_evidence"),
    "execute_information_need_plan": (0.44, "retrieving_evidence"),
    "rerank": (0.56, "reranking_evidence"),
    "validate_information_need_constraints": (0.68, "validating_evidence_constraints"),
    "grade_evidence": (0.78, "grading_evidence"),
    "grade_information_need": (0.78, "grading_evidence"),
    "decide_information_need": (0.87, "deciding_retrieval_retry"),
    "detect_primary_document": (0.93, "selecting_primary_document"),
    "complete_information_need": (1.00, "completing_information_need"),
}


_NODE_PROGRESS: dict[str, tuple[float, str]] = {
    "classify_query": (0.10, "classifying_query"),
    "decompose_information_needs": (0.18, "decomposing_information_needs"),
    "initialize_information_need_work": (0.24, "preparing_information_needs"),
    "resolve_information_needs": (0.28, "resolving_information_needs"),
    "aggregate_information_needs": (0.81, "aggregating_information_needs"),
    "arbitrate_final_evidence": (0.85, "arbitrating_final_evidence"),
    "prepare_evidence_context": (0.89, "preparing_answer_context"),
    "generate_answer": (0.92, "generating_answer"),
}


def _result_payload(
    query_run_id: uuid.UUID,
    pipeline_name: str | None,
    query_status: str,
    product_outcome: QueryProductOutcome | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "query_run_id": str(query_run_id),
        "query_status": query_status,
        "pipeline_name": pipeline_name,
    }
    if product_outcome is not None:
        result["product_outcome"] = product_outcome.value
    return result


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Query job payload field {key!r} must be a non-empty string.")
    return value.strip()


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Query job payload field {key!r} must be a positive integer.")
    return value


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
