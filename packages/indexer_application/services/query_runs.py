from __future__ import annotations

import uuid

from packages.indexer_application.commands.execute_query import execute_selected_pipeline
from packages.indexer_application.dto import QueryRunRecord
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.queries import GetQueryRunHandler, GetQueryRunQuery
from packages.rag_core.pipelines import RetrievalPipeline


async def run_query(
    *,
    uow: UnitOfWork,
    pipeline: RetrievalPipeline,
    question: str,
    top_k: int,
    requested_pipeline_name: str | None = None,
) -> QueryRunRecord:
    """Compatibility facade for callers that already selected a pipeline.

    New entry points should use ``ExecuteQueryHandler`` so pipeline selection and
    execution remain one application use case.
    """

    result = await execute_selected_pipeline(
        uow=uow,
        pipeline=pipeline,
        question=question,
        top_k=top_k,
        requested_pipeline_name=requested_pipeline_name,
    )
    return result.query_run


async def get_query_run(*, uow: UnitOfWork, query_run_id: uuid.UUID) -> QueryRunRecord | None:
    """Compatibility facade; new callers should use ``GetQueryRunHandler``."""

    result = await GetQueryRunHandler(uow=uow)(
        GetQueryRunQuery(query_run_id=query_run_id),
    )
    return result.query_run if result is not None else None
