from __future__ import annotations

from app.schemas.jobs import BackgroundJobResponse
from packages.indexer_application.dto import BackgroundJobRecord


def to_background_job_response(job: BackgroundJobRecord) -> BackgroundJobResponse:
    return BackgroundJobResponse(
        id=job.id,
        job_type=job.job_type.value,
        status=job.status.value,
        priority=job.priority,
        payload=job.payload,
        result=job.result,
        progress=job.progress,
        current_stage=job.current_stage,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        dedupe_key=job.dedupe_key,
        scheduled_at=job.scheduled_at,
        heartbeat_at=job.heartbeat_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
