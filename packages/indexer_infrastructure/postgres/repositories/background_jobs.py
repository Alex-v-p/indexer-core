from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobStatus,
    BackgroundJobSubmission,
    BackgroundJobType,
)
from packages.indexer_infrastructure.postgres.models.background_jobs import BackgroundJob


class SqlAlchemyBackgroundJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(self, submission: BackgroundJobSubmission) -> BackgroundJobRecord:
        if submission.dedupe_key:
            existing = await self._find_active_deduplicated(submission.dedupe_key)
            if existing is not None:
                return _to_record(existing)

        scheduled_at = submission.scheduled_at or datetime.now(UTC)
        model = BackgroundJob(
            job_type=submission.job_type,
            status=BackgroundJobStatus.QUEUED,
            priority=submission.priority,
            payload=dict(submission.payload),
            max_attempts=submission.max_attempts,
            dedupe_key=submission.dedupe_key,
            scheduled_at=scheduled_at,
            current_stage="queued",
        )
        if submission.dedupe_key is None:
            self._session.add(model)
            await self._session.flush()
            return _to_record(model)

        try:
            async with self._session.begin_nested():
                self._session.add(model)
                await self._session.flush()
        except IntegrityError:
            existing = await self._find_active_deduplicated(submission.dedupe_key)
            if existing is None:
                raise
            return _to_record(existing)
        return _to_record(model)

    async def get(self, job_id: uuid.UUID) -> BackgroundJobRecord | None:
        model = await self._session.get(BackgroundJob, job_id)
        return _to_record(model) if model is not None else None

    async def list(
        self,
        *,
        limit: int,
        offset: int,
        job_type: BackgroundJobType | None = None,
        status: BackgroundJobStatus | None = None,
    ) -> list[BackgroundJobRecord]:
        statement = select(BackgroundJob).order_by(BackgroundJob.created_at.desc()).offset(offset).limit(limit)
        if job_type is not None:
            statement = statement.where(BackgroundJob.job_type == job_type)
        if status is not None:
            statement = statement.where(BackgroundJob.status == status)
        result = await self._session.execute(statement)
        return [_to_record(model) for model in result.scalars().all()]

    async def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime,
        stale_before: datetime,
    ) -> BackgroundJobRecord | None:
        stale_lease = or_(
            BackgroundJob.heartbeat_at.is_(None),
            BackgroundJob.heartbeat_at < stale_before,
        )
        abandoned_statement = (
            select(BackgroundJob)
            .where(
                BackgroundJob.status == BackgroundJobStatus.RUNNING,
                BackgroundJob.attempts >= BackgroundJob.max_attempts,
                stale_lease,
            )
            .order_by(BackgroundJob.priority.asc(), BackgroundJob.created_at.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        abandoned_result = await self._session.execute(abandoned_statement)
        abandoned = abandoned_result.scalar_one_or_none()
        if abandoned is not None:
            abandoned.locked_at = now
            abandoned.locked_by = worker_id
            abandoned.heartbeat_at = now
            abandoned.current_stage = "lease_expired"
            await self._session.flush()
            return _to_record(abandoned)

        claimable = or_(
            and_(
                BackgroundJob.status == BackgroundJobStatus.QUEUED,
                BackgroundJob.scheduled_at <= now,
            ),
            and_(
                BackgroundJob.status == BackgroundJobStatus.RUNNING,
                stale_lease,
            ),
        )
        statement = (
            select(BackgroundJob)
            .where(claimable, BackgroundJob.attempts < BackgroundJob.max_attempts)
            .order_by(BackgroundJob.priority.asc(), BackgroundJob.scheduled_at.asc(), BackgroundJob.created_at.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        result = await self._session.execute(statement)
        model = result.scalar_one_or_none()
        if model is None:
            return None

        model.status = BackgroundJobStatus.RUNNING
        model.progress = 0.0
        model.error_message = None
        model.attempts += 1
        model.locked_at = now
        model.locked_by = worker_id
        model.heartbeat_at = now
        model.started_at = model.started_at or now
        model.completed_at = None
        model.current_stage = "claimed"
        await self._session.flush()
        return _to_record(model)

    async def heartbeat(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        now: datetime,
    ) -> None:
        model = await self._require_owned_running(job_id, worker_id)
        model.heartbeat_at = now
        await self._session.flush()

    async def update_progress(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        progress: float,
        current_stage: str,
    ) -> None:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("Background job progress must be between 0 and 1.")
        model = await self._require_owned_running(job_id, worker_id)
        model.progress = progress
        model.current_stage = current_stage
        model.heartbeat_at = datetime.now(UTC)
        await self._session.flush()

    async def mark_succeeded(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        result: dict[str, Any],
        completed_at: datetime,
    ) -> BackgroundJobRecord:
        model = await self._require_owned_running(job_id, worker_id)
        model.status = BackgroundJobStatus.SUCCEEDED
        model.result = dict(result)
        model.progress = 1.0
        model.current_stage = "completed"
        model.completed_at = completed_at
        model.heartbeat_at = completed_at
        model.locked_at = None
        model.locked_by = None
        model.error_message = None
        await self._session.flush()
        return _to_record(model)

    async def mark_failed(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        error_message: str,
        failed_at: datetime,
        retry_at: datetime | None,
    ) -> BackgroundJobRecord:
        model = await self._require_owned_running(job_id, worker_id)
        model.error_message = error_message
        model.locked_at = None
        model.locked_by = None
        model.heartbeat_at = failed_at
        if retry_at is not None and model.attempts < model.max_attempts:
            model.status = BackgroundJobStatus.QUEUED
            model.scheduled_at = retry_at
            model.current_stage = "retry_scheduled"
        else:
            model.status = BackgroundJobStatus.FAILED
            model.current_stage = "failed"
            model.completed_at = failed_at
        await self._session.flush()
        return _to_record(model)

    async def _find_active_deduplicated(self, dedupe_key: str) -> BackgroundJob | None:
        result = await self._session.execute(
            select(BackgroundJob)
            .where(
                BackgroundJob.dedupe_key == dedupe_key,
                BackgroundJob.status.in_(
                    (BackgroundJobStatus.QUEUED, BackgroundJobStatus.RUNNING)
                ),
            )
            .order_by(BackgroundJob.created_at.asc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _require_owned_running(self, job_id: uuid.UUID, worker_id: str) -> BackgroundJob:
        statement = (
            select(BackgroundJob)
            .where(
                BackgroundJob.id == job_id,
                BackgroundJob.status == BackgroundJobStatus.RUNNING,
                BackgroundJob.locked_by == worker_id,
            )
            .with_for_update()
        )
        result = await self._session.execute(statement)
        model = result.scalar_one_or_none()
        if model is None:
            raise LookupError(f"Running background job {job_id} is not owned by worker {worker_id!r}.")
        return model


def _to_record(model: BackgroundJob) -> BackgroundJobRecord:
    return BackgroundJobRecord(
        id=model.id,
        job_type=model.job_type,
        status=model.status,
        priority=model.priority,
        payload=dict(model.payload or {}),
        result=dict(model.result or {}),
        progress=float(model.progress),
        current_stage=model.current_stage,
        attempts=model.attempts,
        max_attempts=model.max_attempts,
        dedupe_key=model.dedupe_key,
        scheduled_at=model.scheduled_at,
        locked_at=model.locked_at,
        locked_by=model.locked_by,
        heartbeat_at=model.heartbeat_at,
        started_at=model.started_at,
        completed_at=model.completed_at,
        error_message=model.error_message,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
