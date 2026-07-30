from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.presenters.query import to_query_response
from app.dependencies.application import get_execute_query_handler, get_query_run_handler
from app.schemas.queries import QueryRequest, QueryResponse
from packages.indexer_application.commands import (
    ExecuteQueryCommand,
    ExecuteQueryHandler,
    UnknownQueryPipelineError,
)
from packages.indexer_application.queries import GetQueryRunHandler, GetQueryRunQuery

router = APIRouter(prefix="/queries", tags=["queries"])


@router.post("", response_model=QueryResponse, status_code=status.HTTP_201_CREATED)
async def create_query_run(
    payload: QueryRequest,
    handler: ExecuteQueryHandler = Depends(get_execute_query_handler),
) -> QueryResponse:
    try:
        result = await handler(
            ExecuteQueryCommand(
                question=payload.question,
                top_k=payload.top_k,
                pipeline_name=payload.pipeline_name,
            ),
        )
    except UnknownQueryPipelineError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return to_query_response(result)


@router.get("/{query_run_id}", response_model=QueryResponse)
async def read_query_run(
    query_run_id: uuid.UUID,
    handler: GetQueryRunHandler = Depends(get_query_run_handler),
) -> QueryResponse:
    result = await handler(GetQueryRunQuery(query_run_id=query_run_id))
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query run not found.")
    return to_query_response(result)
