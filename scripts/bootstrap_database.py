from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.dependencies.database import check_database_connection, close_database_connection

logger = logging.getLogger(__name__)


def get_alembic_config() -> Config:
    """Load Alembic config from the repository/container root."""

    config_path = Path("alembic.ini")
    if not config_path.exists():
        raise FileNotFoundError("Could not find alembic.ini from the current working directory.")
    return Config(str(config_path))


async def wait_for_database(settings: Settings) -> None:
    """Wait for PostgreSQL to accept connections before running migrations."""

    last_error: Exception | None = None
    for attempt in range(1, settings.bootstrap_db_max_attempts + 1):
        try:
            await check_database_connection(settings)
            logger.info("Database connection is ready.")
            return
        except Exception as exc:  # pragma: no cover - depends on external PostgreSQL timing
            last_error = exc
            logger.warning(
                "Database is not ready yet. Retrying.",
                extra={
                    "attempt": attempt,
                    "max_attempts": settings.bootstrap_db_max_attempts,
                    "retry_seconds": settings.bootstrap_db_retry_seconds,
                },
            )
            await asyncio.sleep(settings.bootstrap_db_retry_seconds)

    raise RuntimeError("Database did not become ready in time.") from last_error


def run_migrations() -> None:
    """Apply all pending Alembic migrations.

    Alembic is idempotent: on the first startup it creates the schema and stores
    the applied revision in its version table; on later startups it checks the
    current revision and only applies migrations that are still pending.
    """

    alembic_config = get_alembic_config()
    logger.info("Running database bootstrap migrations.")
    command.upgrade(alembic_config, "head")
    logger.info("Database schema is at the latest migration head.")


async def wait_for_database_and_close(settings: Settings) -> None:
    """Wait for database readiness, then dispose the temporary async engine.

    Alembic runs its own async migration environment through a synchronous
    command API. Keeping this readiness check in a separate event loop prevents
    Alembic's env.py from trying to call asyncio.run() inside an already-running
    loop.
    """

    try:
        await wait_for_database(settings)
    finally:
        await close_database_connection()


def bootstrap_database() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("Starting database bootstrap.")
    asyncio.run(wait_for_database_and_close(settings))
    run_migrations()
    logger.info("Database bootstrap completed successfully.")


def main() -> None:
    bootstrap_database()


if __name__ == "__main__":
    main()
