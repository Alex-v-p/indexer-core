from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from packages.indexer_application.ports import StoredDocumentReference
from packages.indexer_infrastructure.postgres.unit_of_work import SqlAlchemyUnitOfWork


@pytest.mark.asyncio
async def test_split_repositories_share_one_real_postgres_transaction() -> None:
    """Exercise both aggregate repositories against an explicitly configured migrated test DB."""

    database_url = os.getenv("INDEXER_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set INDEXER_TEST_DATABASE_URL to run PostgreSQL repository integration tests.")

    engine = create_async_engine(database_url, pool_pre_ping=True)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    suffix = uuid.uuid4().hex
    try:
        async with factory() as session:
            transaction = await session.begin()
            uow = SqlAlchemyUnitOfWork(session)
            stored = StoredDocumentReference(
                storage_uri=f"s3://integration/{suffix}.md",
                original_filename=f"{suffix}.md",
                content_type="text/markdown",
                size_bytes=12,
                checksum_sha256=suffix.ljust(64, "0")[:64],
                storage_backend="minio",
                bucket_name="integration",
                object_key=f"{suffix}.md",
            )

            document_id = await uow.documents.create_processing_document(
                stored_file=stored,
                title=f"Integration {suffix}",
            )
            version = await uow.documents.create_processing_version(
                document_id=document_id,
                stored_file=stored,
            )
            query_run_id = await uow.query_runs.create_running(
                question="Does the split share a transaction?",
                pipeline_name="integration",
                pipeline_version="1",
                top_k=1,
                requested_pipeline_name=None,
            )
            await uow.flush()

            document = await uow.documents.get(document_id)
            query_run = await uow.query_runs.get(query_run_id)

            assert document is not None
            assert document.versions[0].id == version.id
            assert query_run is not None
            assert query_run.question == "Does the split share a transaction?"
            await transaction.rollback()
    finally:
        await engine.dispose()
