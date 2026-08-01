from __future__ import annotations

import uuid
from pathlib import Path
from datetime import UTC, datetime

from packages.indexer_application.dto import (
    DocumentStatus,
    DocumentVersionStatus,
    QueryRunStatus,
    TraceStepStatus,
)
from packages.indexer_infrastructure.postgres.models import (
    BackgroundJob,
    Citation,
    Document,
    DocumentVersion,
    Evidence,
    QdrantChunkIndex,
    QueryRun,
    TraceStep,
)
from packages.indexer_infrastructure.postgres.repositories import (
    SqlAlchemyBackgroundJobRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyQueryRunRepository,
    SqlAlchemySubjectRepository,
    to_document_record,
    to_query_run_record,
)
from packages.indexer_infrastructure.postgres.repositories.background_jobs import (
    SqlAlchemyBackgroundJobRepository as SplitBackgroundJobRepository,
)
from packages.indexer_infrastructure.postgres.repositories.documents import (
    SqlAlchemyDocumentRepository as SplitDocumentRepository,
)
from packages.indexer_infrastructure.postgres.repositories.query_runs import (
    SqlAlchemyQueryRunRepository as SplitQueryRunRepository,
)
from packages.indexer_infrastructure.postgres.unit_of_work import SqlAlchemyUnitOfWork


def test_repository_package_preserves_legacy_exports() -> None:
    assert SqlAlchemyBackgroundJobRepository is SplitBackgroundJobRepository
    assert SqlAlchemyDocumentRepository is SplitDocumentRepository
    assert SqlAlchemyQueryRunRepository is SplitQueryRunRepository


class RecordingSession:
    def __init__(self) -> None:
        self.flush_calls = 0
        self.commit_calls = 0

    async def flush(self) -> None:
        self.flush_calls += 1

    async def commit(self) -> None:
        self.commit_calls += 1


def test_unit_of_work_composes_aggregate_repositories_without_implicit_commit() -> None:
    session = RecordingSession()
    uow = SqlAlchemyUnitOfWork(session)  # type: ignore[arg-type]

    assert isinstance(uow.background_jobs, SqlAlchemyBackgroundJobRepository)
    assert isinstance(uow.documents, SqlAlchemyDocumentRepository)
    assert isinstance(uow.query_runs, SqlAlchemyQueryRunRepository)
    assert uow.background_jobs._session is session
    assert uow.documents._session is session
    assert uow.query_runs._session is session
    assert isinstance(uow.subjects, SqlAlchemySubjectRepository)
    assert uow.subjects._session is session
    assert session.commit_calls == 0


def test_background_job_model_uses_durable_queue_table() -> None:
    assert BackgroundJob.__tablename__ == "background_jobs"
    index_names = {index.name for index in BackgroundJob.__table__.indexes}
    assert "ix_background_jobs_status_scheduled" in index_names
    assert "uq_background_jobs_active_dedupe_key" in index_names


def test_document_mapper_keeps_nested_versions_and_chunk_indexes() -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    document = Document(
        id=document_id,
        title="Architecture",
        original_filename="architecture.md",
        content_type="text/markdown",
        storage_uri="s3://documents/architecture.md",
        size_bytes=128,
        checksum_sha256="abc",
        status=DocumentStatus.READY,
        metadata_={"latest_version_number": 1},
        created_at=now,
        updated_at=now,
    )
    document.versions = [
        DocumentVersion(
            id=version_id,
            document_id=document_id,
            version_number=1,
            storage_uri=document.storage_uri,
            content_type=document.content_type,
            checksum_sha256=document.checksum_sha256,
            parser_name="markdown",
            parser_version="1",
            status=DocumentVersionStatus.READY,
            metadata_={"chunk_count": 1},
            created_at=now,
            updated_at=now,
        )
    ]
    document.qdrant_chunk_indexes = [
        QdrantChunkIndex(
            id=chunk_id,
            document_id=document_id,
            document_version_id=version_id,
            ordinal=0,
            content_hash="hash",
            token_count=12,
            source_page_start=1,
            source_page_end=1,
            section_title="Architecture",
            qdrant_collection="chunks",
            qdrant_point_id="point-1",
            metadata_={"source": "test"},
            created_at=now,
        )
    ]

    record = to_document_record(document)

    assert record.id == document_id
    assert record.status == DocumentStatus.READY
    assert record.versions[0].id == version_id
    assert record.versions[0].metadata == {"chunk_count": 1}
    assert record.chunk_indexes[0].id == chunk_id
    assert record.chunk_indexes[0].qdrant_point_id == "point-1"


def test_query_run_mapper_keeps_evidence_citations_and_trace() -> None:
    now = datetime(2026, 7, 30, tzinfo=UTC)
    query_run_id = uuid.uuid4()
    evidence_id = uuid.uuid4()
    query_run = QueryRun(
        id=query_run_id,
        question="What changed?",
        answer="The repository was split.",
        status=QueryRunStatus.SUCCEEDED,
        pipeline_name="baseline",
        pipeline_version="1",
        top_k=5,
        started_at=now,
        completed_at=now,
        metadata_={"runner": "graph"},
    )
    query_run.evidence_items = [
        Evidence(
            id=evidence_id,
            query_run_id=query_run_id,
            rank=1,
            score=0.9,
            text="Repository persistence is aggregate-oriented.",
            metadata_={"kind": "source"},
        )
    ]
    query_run.citations = [
        Citation(
            id=uuid.uuid4(),
            query_run_id=query_run_id,
            evidence_id=evidence_id,
            citation_index=1,
            label="[1]",
            quote="aggregate-oriented",
            metadata_={},
        )
    ]
    query_run.trace_steps = [
        TraceStep(
            id=uuid.uuid4(),
            query_run_id=query_run_id,
            step_order=1,
            name="retrieve",
            step_type="retrieval",
            status=TraceStepStatus.SUCCEEDED,
            duration_ms=4,
            metadata_={},
        )
    ]

    record = to_query_run_record(query_run)

    assert record.id == query_run_id
    assert record.status == QueryRunStatus.SUCCEEDED
    assert record.evidence_items[0].id == evidence_id
    assert record.citations[0].evidence_id == evidence_id
    assert record.trace_steps[0].name == "retrieve"


def test_repository_adapters_do_not_own_transaction_commits() -> None:
    repository_root = (
        Path(__file__).resolve().parents[3]
        / "packages"
        / "indexer_infrastructure"
        / "postgres"
        / "repositories"
    )
    offenders = [
        source.relative_to(repository_root)
        for source in repository_root.rglob("*.py")
        if ".commit(" in source.read_text(encoding="utf-8")
    ]

    assert offenders == []
