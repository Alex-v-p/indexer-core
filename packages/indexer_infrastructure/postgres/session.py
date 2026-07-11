from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)


class PostgresSessionManager:
    """Own the reusable SQLAlchemy engine and session factory for one runtime."""

    def __init__(
        self,
        *,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ) -> None:
        self._database_url = database_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._echo = echo
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(
                self._database_url,
                echo=self._echo,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_pre_ping=True,
            )
            logger.info("Database engine initialized.")
        return self._engine

    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                bind=self.engine(),
                expire_on_commit=False,
                autoflush=False,
            )
        return self._session_factory

    async def check_connection(self) -> bool:
        async with self.engine().connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Database engine disposed.")
        self._engine = None
        self._session_factory = None
