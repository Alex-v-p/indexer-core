from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import QueryExecutionResult, QueryRunRecord
from packages.indexer_application.ports import UnitOfWork
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.pipelines import (
    PipelineRegistry,
    RetrievalPipeline,
    UnknownPipelineError,
)


class UnknownQueryPipelineError(ValueError):
    """Raised when a query requests a pipeline that is not registered."""


@dataclass(frozen=True, slots=True)
class ExecuteQueryCommand:
    question: str
    top_k: int
    pipeline_name: str | None = None


class ExecuteQueryHandler:
    """Application entry point for selecting, executing, and persisting a query."""

    def __init__(self, *, uow: UnitOfWork, pipeline_registry: PipelineRegistry) -> None:
        self._uow = uow
        self._pipeline_registry = pipeline_registry

    async def __call__(self, command: ExecuteQueryCommand) -> QueryExecutionResult:
        try:
            pipeline = self._pipeline_registry.build(command.pipeline_name)
        except UnknownPipelineError as exc:
            raise UnknownQueryPipelineError(str(exc)) from exc

        return await execute_selected_pipeline(
            uow=self._uow,
            pipeline=pipeline,
            question=command.question,
            top_k=command.top_k,
            requested_pipeline_name=command.pipeline_name,
        )


async def execute_selected_pipeline(
    *,
    uow: UnitOfWork,
    pipeline: RetrievalPipeline,
    question: str,
    top_k: int,
    requested_pipeline_name: str | None = None,
) -> QueryExecutionResult:
    """Execute an already selected pipeline behind the application command boundary.

    This compatibility seam is intentionally narrower than ``ExecuteQueryHandler``;
    API routes and future workers should use the handler so pipeline selection is not
    duplicated outside the application layer.
    """

    query_run_id = await uow.query_runs.create_running(
        question=question,
        pipeline_name=pipeline.name,
        pipeline_version=pipeline.version,
        top_k=top_k,
        requested_pipeline_name=requested_pipeline_name,
    )
    state = QueryState(
        question=question,
        top_k=top_k,
        query_run_id=query_run_id,
        requested_pipeline_name=requested_pipeline_name,
        pipeline_name=pipeline.name,
        pipeline_version=pipeline.version,
    )

    try:
        state = await pipeline.run(state)
    except Exception as exc:
        await uow.query_runs.mark_failed(
            query_run_id=query_run_id,
            error_message=str(exc),
            trace=state.trace,
        )
        await uow.commit()
        return QueryExecutionResult.from_record(
            await _require_query_run(uow, query_run_id),
        )

    await uow.query_runs.mark_succeeded(query_run_id=query_run_id, state=state)
    await uow.commit()
    return QueryExecutionResult.from_record(
        await _require_query_run(uow, query_run_id),
    )


async def _require_query_run(uow: UnitOfWork, query_run_id: uuid.UUID) -> QueryRunRecord:
    query_run = await uow.query_runs.get(query_run_id)
    if query_run is None:
        raise LookupError(f"Query run {query_run_id} could not be reloaded.")
    return query_run
