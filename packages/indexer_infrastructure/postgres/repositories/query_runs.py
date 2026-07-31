from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.indexer_application.dto import QueryRunRecord, QueryRunStatus, TraceStepStatus
from packages.indexer_infrastructure.postgres.models import (
    Citation,
    Evidence,
    QueryRun,
    TraceStep,
)
from packages.indexer_infrastructure.postgres.repositories.mappers import to_query_run_record
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.runtime import TraceEvent


class SqlAlchemyQueryRunRepository:
    """SQLAlchemy adapter for query executions and their evidence aggregate."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_pending(
        self,
        *,
        question: str,
        pipeline_name: str,
        pipeline_version: str,
        top_k: int,
        requested_pipeline_name: str | None,
    ) -> uuid.UUID:
        query_run = QueryRun(
            question=question,
            status=QueryRunStatus.PENDING,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            top_k=top_k,
            metadata_={
                "runner": "background_job",
                "requested_pipeline_name": requested_pipeline_name,
            },
        )
        self._session.add(query_run)
        await self._session.flush()
        return query_run.id

    async def create_running(
        self,
        *,
        question: str,
        pipeline_name: str,
        pipeline_version: str,
        top_k: int,
        requested_pipeline_name: str | None,
    ) -> uuid.UUID:
        query_run = QueryRun(
            question=question,
            status=QueryRunStatus.RUNNING,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            top_k=top_k,
            metadata_={"runner": "graph", "requested_pipeline_name": requested_pipeline_name},
        )
        self._session.add(query_run)
        await self._session.flush()
        return query_run.id

    async def mark_running(
        self,
        *,
        query_run_id: uuid.UUID,
        pipeline_name: str,
        pipeline_version: str,
        background_job_id: uuid.UUID,
        attempt: int,
    ) -> None:
        query_run = await self._require_query_run(query_run_id)
        query_run.status = QueryRunStatus.RUNNING
        query_run.pipeline_name = pipeline_name
        query_run.pipeline_version = pipeline_version
        query_run.started_at = datetime.now(UTC)
        query_run.completed_at = None
        query_run.error_message = None
        query_run.metadata_ = {
            **(query_run.metadata_ or {}),
            "runner": "background_job",
            "background_job_id": str(background_job_id),
            "background_job_attempt": attempt,
        }

    async def mark_retry_pending(
        self,
        *,
        query_run_id: uuid.UUID,
        error_message: str,
        retry_at: datetime,
    ) -> None:
        query_run = await self._require_query_run(query_run_id)
        if query_run.status is QueryRunStatus.SUCCEEDED:
            return
        query_run.status = QueryRunStatus.PENDING
        query_run.completed_at = None
        query_run.error_message = error_message
        query_run.metadata_ = {
            **(query_run.metadata_ or {}),
            "retry_scheduled_at": retry_at.isoformat(),
            "last_attempt_error": error_message,
        }

    async def mark_failed(
        self,
        *,
        query_run_id: uuid.UUID,
        error_message: str,
        trace: list[TraceEvent],
    ) -> None:
        query_run = await self._require_query_run(query_run_id)
        if query_run.status is QueryRunStatus.SUCCEEDED:
            return
        query_run.status = QueryRunStatus.FAILED
        query_run.completed_at = datetime.now(UTC)
        query_run.error_message = error_message
        if trace:
            await self._replace_trace_steps(query_run_id, trace)

    async def mark_succeeded(self, *, query_run_id: uuid.UUID, state: QueryState) -> None:
        query_run = await self._require_query_run(query_run_id)
        query_run.answer = state.answer
        query_run.status = QueryRunStatus.SUCCEEDED
        query_run.completed_at = datetime.now(UTC)
        query_run.error_message = None
        query_run.pipeline_name = state.pipeline_name
        query_run.pipeline_version = state.pipeline_version
        query_run.metadata_ = {**(query_run.metadata_ or {}), **state.metadata}

        await self._clear_result_children(query_run_id)
        evidence_by_rank: dict[int, Evidence] = {}
        for item in state.retrieved_evidence:
            evidence = Evidence(
                query_run_id=query_run_id,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                rank=item.rank,
                score=item.score,
                text=item.text,
                metadata_=item.metadata,
            )
            self._session.add(evidence)
            evidence_by_rank[item.rank] = evidence
        await self._session.flush()

        for item in state.citations:
            evidence = evidence_by_rank.get(item.evidence_rank or -1)
            self._session.add(
                Citation(
                    query_run_id=query_run_id,
                    evidence_id=evidence.id if evidence else None,
                    document_id=item.document_id,
                    document_version_id=item.document_version_id,
                    qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                    citation_index=item.citation_index,
                    label=item.label,
                    page_number=item.page_number,
                    quote=item.quote,
                    metadata_=item.metadata,
                ),
            )
        self._add_trace_steps(query_run_id, state.trace)

    async def get(self, query_run_id: uuid.UUID) -> QueryRunRecord | None:
        statement = (
            select(QueryRun)
            .where(QueryRun.id == query_run_id)
            .options(
                selectinload(QueryRun.evidence_items),
                selectinload(QueryRun.citations),
                selectinload(QueryRun.trace_steps),
            )
        )
        result = await self._session.execute(statement)
        model = result.scalar_one_or_none()
        return to_query_run_record(model) if model else None

    async def _require_query_run(self, query_run_id: uuid.UUID) -> QueryRun:
        query_run = await self._session.get(QueryRun, query_run_id)
        if query_run is None:
            raise LookupError(f"Query run {query_run_id} was not found.")
        return query_run

    async def _clear_result_children(self, query_run_id: uuid.UUID) -> None:
        await self._session.execute(delete(Citation).where(Citation.query_run_id == query_run_id))
        await self._session.execute(delete(Evidence).where(Evidence.query_run_id == query_run_id))
        await self._session.execute(delete(TraceStep).where(TraceStep.query_run_id == query_run_id))

    async def _replace_trace_steps(self, query_run_id: uuid.UUID, trace: list[TraceEvent]) -> None:
        await self._session.execute(delete(TraceStep).where(TraceStep.query_run_id == query_run_id))
        self._add_trace_steps(query_run_id, trace)

    def _add_trace_steps(self, query_run_id: uuid.UUID, trace: list[TraceEvent]) -> None:
        for item in trace:
            self._session.add(
                TraceStep(
                    query_run_id=query_run_id,
                    step_order=item.step_order,
                    name=item.name,
                    step_type=item.step_type,
                    status=TraceStepStatus(item.status),
                    duration_ms=item.duration_ms,
                    input_summary=item.input_summary,
                    output_summary=item.output_summary,
                    error_message=item.error_message,
                    metadata_=item.metadata,
                ),
            )
