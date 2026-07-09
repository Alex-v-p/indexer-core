import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create or return the shared async database engine."""

    global _engine
    if _engine is None:
        resolved_settings = settings or get_settings()
        _engine = create_async_engine(
            resolved_settings.database_url,
            echo=resolved_settings.db_echo,
            pool_size=resolved_settings.db_pool_size,
            max_overflow=resolved_settings.db_max_overflow,
            pool_pre_ping=True,
        )
        logger.info("Database engine initialized.")
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Create or return the shared async session factory."""

    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(settings),
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides one database session per request."""

    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


async def check_database_connection(settings: Settings | None = None) -> bool:
    """Run a lightweight readiness query against the configured database."""

    engine = get_engine(settings)
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    return True


async def close_database_connection() -> None:
    """Dispose the shared database engine on application shutdown."""

    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database engine disposed.")
    _engine = None
    _session_factory = None
