from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.presenters.background_jobs import to_background_job_response
from app.api.presenters.query import to_query_response
from app.dependencies.application import get_query_run_handler, get_submit_query_handler
from app.schemas.queries import QueryRequest, QueryResponse, QueuedQueryResponse
from packages.indexer_application.commands import (
    SubmitQueryCommand,
    SubmitQueryHandler,
    UnknownQueryPipelineError,
)
from packages.indexer_application.queries import GetQueryRunHandler, GetQueryRunQuery
from packages.rag_core.document_scope import CoverageMode

router = APIRouter(prefix="/queries", tags=["queries"])


@router.post("", response_model=QueuedQueryResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_query_run(
    payload: QueryRequest,
    http_request: Request,
    response: Response,
    handler: SubmitQueryHandler = Depends(get_submit_query_handler),
) -> QueuedQueryResponse:
    try:
        result = await handler(
            SubmitQueryCommand(
                question=payload.question,
                top_k=payload.top_k,
                pipeline_name=payload.pipeline_name,
                scheduled_at=payload.scheduled_at,
                subject_ids=tuple(payload.subject_ids),
                coverage_mode=CoverageMode(payload.coverage_mode),
            ),
        )
    except UnknownQueryPipelineError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    response.headers["Location"] = str(
        http_request.url_for("get_background_job", job_id=str(result.job.id))
    )
    response.headers["Content-Location"] = str(
        http_request.url_for("read_query_run", query_run_id=str(result.query.query_run.id))
    )
    return QueuedQueryResponse(
        query=to_query_response(result.query),
        job=to_background_job_response(result.job),
    )


@router.get("/{query_run_id}", response_model=QueryResponse)
async def read_query_run(
    query_run_id: uuid.UUID,
    handler: GetQueryRunHandler = Depends(get_query_run_handler),
) -> QueryResponse:
    result = await handler(GetQueryRunQuery(query_run_id=query_run_id))
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query run not found.")
    return to_query_response(result)
