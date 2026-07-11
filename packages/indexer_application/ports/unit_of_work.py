from __future__ import annotations

from typing import Protocol

from packages.indexer_application.ports.repositories import DocumentRepository, QueryRunRepository


class UnitOfWork(Protocol):
    """Transaction boundary shared by application services."""

    documents: DocumentRepository
    query_runs: QueryRunRepository

    async def flush(self) -> None: ...

    async def commit(self) -> None: ...
