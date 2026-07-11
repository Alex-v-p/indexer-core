import logging
from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from packages.indexer_infrastructure.postgres.session import PostgresSessionManager
from packages.indexer_infrastructure.postgres.unit_of_work import SqlAlchemyUnitOfWork

logger = logging.getLogger(__name__)

_manager: PostgresSessionManager | None = None


def get_database_manager(settings: Settings | None = None) -> PostgresSessionManager:
    global _manager
    if _manager is None:
        resolved = settings or get_settings()
        _manager = PostgresSessionManager(
            database_url=resolved.database_url,
            pool_size=resolved.db_pool_size,
            max_overflow=resolved.db_max_overflow,
            echo=resolved.db_echo,
        )
    return _manager


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    return get_database_manager(settings).engine()


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    return get_database_manager(settings).session_factory()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


def get_unit_of_work(session: AsyncSession = Depends(get_session)) -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(session)


async def check_database_connection(settings: Settings | None = None) -> bool:
    return await get_database_manager(settings).check_connection()


async def close_database_connection() -> None:
    global _manager
    if _manager is not None:
        await _manager.close()
    _manager = None
