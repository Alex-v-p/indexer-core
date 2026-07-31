from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Protocol

from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobStatus,
    BackgroundJobSubmission,
    BackgroundJobType,
)


class BackgroundJobRepository(Protocol):
    async def enqueue(self, submission: BackgroundJobSubmission) -> BackgroundJobRecord: ...

    async def get(self, job_id: uuid.UUID) -> BackgroundJobRecord | None: ...

    async def list(
        self,
        *,
        limit: int,
        offset: int,
        job_type: BackgroundJobType | None = None,
        status: BackgroundJobStatus | None = None,
    ) -> list[BackgroundJobRecord]: ...

    async def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime,
        stale_before: datetime,
    ) -> BackgroundJobRecord | None: ...

    async def heartbeat(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        now: datetime,
    ) -> None: ...

    async def update_progress(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        progress: float,
        current_stage: str,
    ) -> None: ...

    async def mark_succeeded(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        result: dict[str, Any],
        completed_at: datetime,
    ) -> BackgroundJobRecord: ...

    async def mark_failed(
        self,
        *,
        job_id: uuid.UUID,
        worker_id: str,
        error_message: str,
        failed_at: datetime,
        retry_at: datetime | None,
    ) -> BackgroundJobRecord: ...
