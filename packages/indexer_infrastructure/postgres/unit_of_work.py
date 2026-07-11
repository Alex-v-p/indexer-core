from sqlalchemy.ext.asyncio import AsyncSession

from packages.indexer_infrastructure.postgres.repositories import (
    SqlAlchemyDocumentRepository,
    SqlAlchemyQueryRunRepository,
)


class SqlAlchemyUnitOfWork:
    """Request/job-scoped unit of work backed by an existing async session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.documents = SqlAlchemyDocumentRepository(session)
        self.query_runs = SqlAlchemyQueryRunRepository(session)

    async def flush(self) -> None:
        await self._session.flush()

    async def commit(self) -> None:
        await self._session.commit()
