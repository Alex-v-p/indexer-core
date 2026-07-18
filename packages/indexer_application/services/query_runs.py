from __future__ import annotations

import uuid

from packages.indexer_application.dto import QueryRunRecord
from packages.indexer_application.ports import UnitOfWork
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.pipelines import RetrievalPipeline


async def run_query(
    *,
    uow: UnitOfWork,
    pipeline: RetrievalPipeline,
    question: str,
    top_k: int,
    requested_pipeline_name: str | None = None,
) -> QueryRunRecord:
    """Persist and execute a query through an already selected retrieval pipeline."""

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
        return await _require_query_run(uow, query_run_id)

    await uow.query_runs.mark_succeeded(query_run_id=query_run_id, state=state)
    await uow.commit()
    return await _require_query_run(uow, query_run_id)


async def get_query_run(*, uow: UnitOfWork, query_run_id: uuid.UUID) -> QueryRunRecord | None:
    return await uow.query_runs.get(query_run_id)


async def _require_query_run(uow: UnitOfWork, query_run_id: uuid.UUID) -> QueryRunRecord:
    query_run = await uow.query_runs.get(query_run_id)
    if query_run is None:
        raise LookupError(f"Query run {query_run_id} could not be reloaded.")
    return query_run
