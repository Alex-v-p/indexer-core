from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import BackgroundJobRecord, BackgroundJobStatus, BackgroundJobType
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class GetBackgroundJobQuery:
    job_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class ListBackgroundJobsQuery:
    limit: int = 50
    offset: int = 0
    job_type: BackgroundJobType | None = None
    status: BackgroundJobStatus | None = None


class GetBackgroundJobHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: GetBackgroundJobQuery) -> BackgroundJobRecord | None:
        return await self._uow.background_jobs.get(query.job_id)


class ListBackgroundJobsHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: ListBackgroundJobsQuery) -> list[BackgroundJobRecord]:
        return await self._uow.background_jobs.list(
            limit=min(max(query.limit, 1), 100),
            offset=max(query.offset, 0),
            job_type=query.job_type,
            status=query.status,
        )
