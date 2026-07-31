from __future__ import annotations

from typing import Protocol

from packages.indexer_application.ports.background_jobs import BackgroundJobRepository
from packages.indexer_application.ports.repositories import DocumentRepository, QueryRunRepository


class UnitOfWork(Protocol):
    """Application-owned transaction boundary over aggregate repositories.

    Infrastructure implementations expose ``flush`` and ``commit`` but never
    invoke them implicitly; command handlers and coordinators decide durability.
    """

    documents: DocumentRepository
    query_runs: QueryRunRepository
    background_jobs: BackgroundJobRepository

    async def flush(self) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...
