from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.presenters.background_jobs import to_background_job_response
from app.dependencies.application import (
    get_background_job_handler,
    get_enqueue_evaluation_handler,
    get_list_background_jobs_handler,
)
from app.schemas.jobs import BackgroundJobResponse, EvaluationJobRequest
from packages.indexer_application.commands import EnqueueEvaluationCommand, EnqueueEvaluationHandler
from packages.indexer_application.dto import BackgroundJobStatus, BackgroundJobType
from packages.indexer_application.queries import (
    GetBackgroundJobHandler,
    GetBackgroundJobQuery,
    ListBackgroundJobsHandler,
    ListBackgroundJobsQuery,
)

router = APIRouter(prefix="/jobs", tags=["background jobs"])


@router.get("", response_model=list[BackgroundJobResponse])
async def list_background_jobs(
    limit: int = 50,
    offset: int = 0,
    job_type: BackgroundJobType | None = None,
    job_status: BackgroundJobStatus | None = None,
    handler: ListBackgroundJobsHandler = Depends(get_list_background_jobs_handler),
) -> list[BackgroundJobResponse]:
    jobs = await handler(
        ListBackgroundJobsQuery(
            limit=limit,
            offset=offset,
            job_type=job_type,
            status=job_status,
        ),
    )
    return [to_background_job_response(job) for job in jobs]


@router.get("/{job_id}", response_model=BackgroundJobResponse)
async def get_background_job(
    job_id: uuid.UUID,
    handler: GetBackgroundJobHandler = Depends(get_background_job_handler),
) -> BackgroundJobResponse:
    job = await handler(GetBackgroundJobQuery(job_id=job_id))
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Background job not found.")
    return to_background_job_response(job)


@router.post(
    "/evaluations",
    response_model=BackgroundJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_evaluation(
    request: EvaluationJobRequest,
    http_request: Request,
    response: Response,
    handler: EnqueueEvaluationHandler = Depends(get_enqueue_evaluation_handler),
) -> BackgroundJobResponse:
    try:
        job = await handler(
            EnqueueEvaluationCommand(
                dataset_path=request.dataset_path,
                pipeline_name=request.pipeline_name,
                top_k=request.top_k,
                repetitions=request.repetitions,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    response.headers["Location"] = str(
        http_request.url_for("get_background_job", job_id=str(job.id))
    )
    return to_background_job_response(job)
