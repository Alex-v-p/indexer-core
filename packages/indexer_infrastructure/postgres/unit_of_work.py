from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from packages.indexer_infrastructure.postgres.repositories import (
    SqlAlchemyBackgroundJobRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyQueryRunRepository,
)


class SqlAlchemyUnitOfWork:
    """Expose aggregate repositories over one application-owned transaction.

    The unit of work never commits implicitly. Application command handlers or
    coordinators call ``commit`` after they have ordered all PostgreSQL and
    external-system operations for the use case.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.documents = SqlAlchemyDocumentRepository(session)
        self.query_runs = SqlAlchemyQueryRunRepository(session)
        self.background_jobs = SqlAlchemyBackgroundJobRepository(session)

    async def flush(self) -> None:
        await self._session.flush()

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()
