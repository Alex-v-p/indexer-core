from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.database import close_database_connection


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Application lifespan hooks.

    The database engine is created lazily by database dependencies and the
    readiness endpoint. This keeps the API container bootable while readiness
    reports whether PostgreSQL is actually available.
    """

    yield
    await close_database_connection()
