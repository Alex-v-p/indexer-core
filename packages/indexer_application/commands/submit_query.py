from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from packages.indexer_application.commands.execute_query import UnknownQueryPipelineError
from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobSubmission,
    BackgroundJobType,
    QueryExecutionResult,
)
from packages.indexer_application.ports import UnitOfWork
from packages.rag_core.pipelines import PipelineRegistry, UnknownPipelineError
from packages.rag_core.document_scope import CoverageMode
import uuid


@dataclass(frozen=True, slots=True)
class SubmitQueryCommand:
    question: str
    top_k: int
    pipeline_name: str | None = None
    scheduled_at: datetime | None = None
    subject_ids: tuple[uuid.UUID, ...] = ()
    coverage_mode: CoverageMode = CoverageMode.BEST_EVIDENCE


@dataclass(frozen=True, slots=True)
class QueuedQueryResult:
    query: QueryExecutionResult
    job: BackgroundJobRecord


class SubmitQueryHandler:
    """Create a pending query run and enqueue its complete graph execution."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        pipeline_registry: PipelineRegistry,
        max_attempts: int = 2,
        priority: int = 25,
    ) -> None:
        self._uow = uow
        self._pipeline_registry = pipeline_registry
        self._max_attempts = max_attempts
        self._priority = priority

    async def __call__(self, command: SubmitQueryCommand) -> QueuedQueryResult:
        question = command.question.strip()
        if not question:
            raise ValueError("question must not be empty.")
        if command.top_k <= 0:
            raise ValueError("top_k must be positive.")
        scheduled_at = _normalize_schedule(command.scheduled_at)

        try:
            pipeline = self._pipeline_registry.build(command.pipeline_name)
        except UnknownPipelineError as exc:
            raise UnknownQueryPipelineError(str(exc)) from exc

        query_run_id = await self._uow.query_runs.create_pending(
            question=question,
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
            top_k=command.top_k,
            requested_pipeline_name=command.pipeline_name,
            requested_subject_ids=tuple(dict.fromkeys(command.subject_ids)),
            coverage_mode=command.coverage_mode.value,
        )
        job = await self._uow.background_jobs.enqueue(
            BackgroundJobSubmission(
                job_type=BackgroundJobType.RUN_QUERY,
                payload={
                    "query_run_id": str(query_run_id),
                    "pipeline_name": pipeline.name,
                    "requested_pipeline_name": command.pipeline_name,
                    "top_k": command.top_k,
                    "subject_ids": [str(item) for item in dict.fromkeys(command.subject_ids)],
                    "coverage_mode": command.coverage_mode.value,
                },
                priority=self._priority,
                max_attempts=self._max_attempts,
                scheduled_at=scheduled_at,
            ),
        )
        await self._uow.commit()

        query_run = await self._uow.query_runs.get(query_run_id)
        if query_run is None:
            raise LookupError(f"Queued query run {query_run_id} could not be reloaded.")
        return QueuedQueryResult(
            query=QueryExecutionResult.from_record(query_run),
            job=job,
        )


def _normalize_schedule(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scheduled_at must include a timezone offset.")
    return value.astimezone(UTC)
