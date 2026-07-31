from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi.testclient import TestClient
import pytest

from app.core.config import Settings
from app.dependencies.application import (
    get_background_job_handler,
    get_enqueue_document_version_deletion_handler,
    get_enqueue_evaluation_handler,
    get_list_background_jobs_handler,
    get_submit_document_ingestion_handler,
)
from app.main import create_app
from packages.indexer_application.commands import (
    DocumentVersionDeletionTarget,
    EnqueueDocumentVersionDeletionCommand,
    EnqueueDocumentVersionDeletionHandler,
    EnqueueDocumentMaintenanceCommand,
    EnqueueDocumentMaintenanceHandler,
    QueuedDocumentIngestionResult,
    SubmitDocumentIngestionCommand,
    SubmitDocumentIngestionHandler,
)
from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobStatus,
    BackgroundJobSubmission,
    BackgroundJobType,
    DocumentRecord,
    DocumentStatus,
    DocumentVersionDeletionOutcome,
    DocumentVersionIdentity,
    DocumentVersionRecord,
    DocumentVersionStatus,
)
from packages.indexer_application.ports import StoredDocumentReference
from packages.indexer_application.services.background_jobs import (
    DeleteDocumentVersionsJobHandler,
    prepared_document_from_payload,
    prepared_document_to_payload,
)
from packages.indexer_application.services.chunk_indexing import (
    chunk_index_id,
    chunk_point_id,
)
from packages.indexer_application.services.ingestion.prepare import PreparedDocument
from packages.rag_core.documents import UnsupportedDocumentTypeError


NOW = datetime(2026, 7, 31, 9, 30, tzinfo=UTC)


def _job(
    *,
    job_type: BackgroundJobType = BackgroundJobType.INGEST_DOCUMENT,
    payload: dict[str, object] | None = None,
) -> BackgroundJobRecord:
    return BackgroundJobRecord(
        id=uuid.uuid4(),
        job_type=job_type,
        status=BackgroundJobStatus.QUEUED,
        priority=100,
        payload=dict(payload or {}),
        result={},
        progress=0.0,
        current_stage="queued",
        attempts=0,
        max_attempts=3,
        dedupe_key=None,
        scheduled_at=NOW,
        locked_at=None,
        locked_by=None,
        heartbeat_at=None,
        started_at=None,
        completed_at=None,
        error_message=None,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeUpload:
    filename = "architecture.md"
    content_type = "text/markdown"

    async def read(self, size: int = -1) -> bytes:
        del size
        return b"# Architecture"


class FakeObjectStore:
    def __init__(self) -> None:
        self.saved_upload = None
        self.deleted_references: list[StoredDocumentReference] = []
        self.reference = StoredDocumentReference(
            storage_uri="minio://documents/architecture.md",
            original_filename="architecture.md",
            content_type="text/markdown",
            size_bytes=14,
            checksum_sha256="abc123",
            storage_backend="minio",
            bucket_name="documents",
            object_key="architecture.md",
        )

    async def save_upload(self, upload) -> StoredDocumentReference:
        self.saved_upload = upload
        return self.reference

    async def delete(self, reference: StoredDocumentReference) -> None:
        self.deleted_references.append(reference)


class FakeDocumentRepository:
    def __init__(self) -> None:
        self.document_id = uuid.uuid4()
        self.version = DocumentVersionIdentity(
            id=uuid.uuid4(),
            version_number=1,
            uploaded_at=NOW,
        )

    async def find_version_candidate(self, **kwargs):
        self.find_kwargs = kwargs
        return None

    async def create_processing_document(self, **kwargs) -> uuid.UUID:
        self.create_document_kwargs = kwargs
        return self.document_id

    async def create_processing_version(self, **kwargs) -> DocumentVersionIdentity:
        self.create_version_kwargs = kwargs
        return self.version

    async def delete(self, *, document_id: uuid.UUID) -> bool:
        if document_id != self.document_id:
            return False
        self.deleted_document_id = document_id
        return True

    async def get(self, document_id: uuid.UUID) -> DocumentRecord | None:
        if document_id != self.document_id:
            return None
        return DocumentRecord(
            id=self.document_id,
            title="Architecture",
            original_filename="architecture.md",
            content_type="text/markdown",
            storage_uri="minio://documents/architecture.md",
            size_bytes=14,
            checksum_sha256="abc123",
            status=DocumentStatus.PROCESSING,
            metadata={},
            created_at=NOW,
            updated_at=NOW,
            versions=(
                DocumentVersionRecord(
                    id=self.version.id,
                    version_number=1,
                    storage_uri="minio://documents/architecture.md",
                    content_type="text/markdown",
                    checksum_sha256="abc123",
                    parser_name=None,
                    parser_version=None,
                    status=DocumentVersionStatus.PROCESSING,
                    metadata={},
                    created_at=NOW,
                    updated_at=NOW,
                ),
            ),
        )


class FakeBackgroundJobRepository:
    def __init__(self) -> None:
        self.submissions: list[BackgroundJobSubmission] = []
        self.active_for_document = False

    async def has_active_for_document(self, **kwargs) -> bool:
        self.active_check = kwargs
        return self.active_for_document

    async def enqueue(self, submission: BackgroundJobSubmission) -> BackgroundJobRecord:
        self.submissions.append(submission)
        return _job(job_type=submission.job_type, payload=submission.payload)


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.documents = FakeDocumentRepository()
        self.background_jobs = FakeBackgroundJobRepository()
        self.commit_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1


async def test_submit_document_ingestion_persists_source_and_only_enqueues_heavy_work() -> None:
    uow = FakeUnitOfWork()
    object_store = FakeObjectStore()
    upload = FakeUpload()
    handler = SubmitDocumentIngestionHandler(
        uow=uow,  # type: ignore[arg-type]
        object_store=object_store,  # type: ignore[arg-type]
        max_attempts=4,
    )

    result = await handler(
        SubmitDocumentIngestionCommand(
            upload=upload,
            title="Architecture",
        ),
    )

    assert object_store.saved_upload is upload
    assert result.document.status is DocumentStatus.PROCESSING
    assert result.job.status is BackgroundJobStatus.QUEUED
    assert uow.commit_calls == 1
    submission = uow.background_jobs.submissions[0]
    assert submission.job_type is BackgroundJobType.INGEST_DOCUMENT
    assert submission.max_attempts == 4
    assert submission.dedupe_key == f"ingestion:{uow.documents.version.id}"
    assert submission.payload["document_version_id"] == str(uow.documents.version.id)
    stored_payload = submission.payload["stored_document"]
    assert isinstance(stored_payload, dict)
    assert stored_payload["storage_uri"] == object_store.reference.storage_uri


async def test_document_maintenance_targets_latest_ready_version() -> None:
    uow = FakeUnitOfWork()
    older = _version(number=1)
    latest = _version(number=2)

    async def get_document(document_id: uuid.UUID) -> DocumentRecord:
        return DocumentRecord(
            id=document_id,
            title="Architecture",
            original_filename="architecture-v2.md",
            content_type="text/markdown",
            storage_uri=latest.storage_uri,
            size_bytes=20,
            checksum_sha256="v2",
            status=DocumentStatus.READY,
            metadata={},
            created_at=NOW,
            updated_at=NOW,
            versions=(older, latest),
        )

    uow.documents.get = get_document  # type: ignore[method-assign]
    handler = EnqueueDocumentMaintenanceHandler(uow=uow, max_attempts=2)  # type: ignore[arg-type]

    job = await handler(
        EnqueueDocumentMaintenanceCommand(
            document_id=uow.documents.document_id,
            contextualization_only=True,
        ),
    )

    assert job is not None
    submission = uow.background_jobs.submissions[0]
    assert submission.job_type is BackgroundJobType.CONTEXTUALIZE_DOCUMENT
    assert submission.payload["document_version_id"] == str(latest.id)
    assert submission.dedupe_key == f"contextualize_document:{latest.id}"
    assert submission.max_attempts == 2


def _version(*, number: int) -> DocumentVersionRecord:
    return DocumentVersionRecord(
        id=uuid.uuid4(),
        version_number=number,
        storage_uri=f"s3://documents/v{number}.md",
        content_type="text/markdown",
        checksum_sha256=f"v{number}",
        parser_name="markdown",
        parser_version="1",
        status=DocumentVersionStatus.READY,
        metadata={},
        created_at=NOW,
        updated_at=NOW,
    )



async def test_document_version_deletion_batches_explicit_versions_when_document_is_idle() -> None:
    uow = FakeUnitOfWork()
    handler = EnqueueDocumentVersionDeletionHandler(
        uow=uow,  # type: ignore[arg-type]
        max_attempts=4,
    )
    target = DocumentVersionDeletionTarget(
        document_id=uow.documents.document_id,
        version_id=uow.documents.version.id,
    )

    job = await handler(EnqueueDocumentVersionDeletionCommand(targets=(target, target)))

    submission = uow.background_jobs.submissions[0]
    assert job.job_type is BackgroundJobType.DELETE_DOCUMENT_VERSIONS
    assert submission.job_type is BackgroundJobType.DELETE_DOCUMENT_VERSIONS
    assert submission.max_attempts == 4
    assert submission.dedupe_key is not None
    assert submission.dedupe_key.startswith("delete_document_versions:")
    assert submission.payload["targets"] == [
        {
            "document_id": str(uow.documents.document_id),
            "document_title": "Architecture",
            "document_version_id": str(uow.documents.version.id),
            "document_version_number": 1,
        }
    ]
    assert uow.commit_calls == 1

    uow.background_jobs.active_for_document = True
    with pytest.raises(ValueError, match="active background work"):
        await handler(EnqueueDocumentVersionDeletionCommand(targets=(target,)))


async def test_delete_document_versions_job_removes_only_target_and_promotes_remaining_version() -> None:
    older = _version(number=1)
    latest = _version(number=2)
    document_id = uuid.uuid4()
    document = DocumentRecord(
        id=document_id,
        title="Architecture",
        original_filename="architecture-v2.md",
        content_type="text/markdown",
        storage_uri=latest.storage_uri,
        size_bytes=20,
        checksum_sha256=latest.checksum_sha256,
        status=DocumentStatus.READY,
        metadata={},
        created_at=NOW,
        updated_at=NOW,
        versions=(older, latest),
    )

    class Documents:
        async def get(self, requested_id):
            return document if requested_id == document_id else None

        async def delete_version(self, *, document_id: uuid.UUID, version_id: uuid.UUID):
            assert document_id == document.id
            assert version_id == latest.id
            return DocumentVersionDeletionOutcome(
                deleted=True,
                document_deleted=False,
                promoted_version_id=older.id,
                promoted_version_number=older.version_number,
            )

    class Uow:
        documents = Documents()

    class Index:
        deleted: list[tuple[uuid.UUID, uuid.UUID]] = []
        promoted: list[tuple[uuid.UUID, uuid.UUID]] = []

        async def delete_document_version(
            self,
            *,
            document_id: uuid.UUID,
            version_id: uuid.UUID,
        ) -> None:
            self.deleted.append((document_id, version_id))

        async def activate_document_version(
            self,
            *,
            document_id: uuid.UUID,
            version_id: uuid.UUID,
        ) -> None:
            self.promoted.append((document_id, version_id))

    class Cache:
        calls = 0

        def invalidate(self) -> None:
            self.calls += 1

    store = FakeObjectStore()
    index = Index()
    cache = Cache()
    stages: list[str] = []

    async def report(progress: float, stage: str) -> None:
        assert 0 <= progress <= 1
        stages.append(stage)

    result = await DeleteDocumentVersionsJobHandler(
        uow=Uow(),  # type: ignore[arg-type]
        object_store=store,
        document_index=index,
        version_index=index,
        keyword_cache=cache,
    )(
        {
            "targets": [
                {
                    "document_id": str(document_id),
                    "document_version_id": str(latest.id),
                }
            ]
        },
        report,
    )

    assert index.deleted == [(document_id, latest.id)]
    assert index.promoted == [(document_id, older.id)]
    assert [item.storage_uri for item in store.deleted_references] == [latest.storage_uri]
    assert cache.calls == 1
    assert result["deleted_version_count"] == 1
    assert result["deleted_document_ids"] == []
    assert stages == [
        "loading_document_version_for_deletion",
        "removing_version_vectors",
        "removing_version_source_file",
        "updating_document_version_records",
        "promoting_remaining_document_version",
        "invalidating_keyword_index",
        "document_version_deletion_complete",
    ]


def test_prepared_document_job_payload_round_trips_durable_reference() -> None:
    prepared = PreparedDocument(
        stored_document=FakeObjectStore().reference,
        document_id=uuid.uuid4(),
        document_title="Architecture",
        version=DocumentVersionIdentity(
            id=uuid.uuid4(),
            version_number=3,
            uploaded_at=NOW,
            published_at=NOW,
        ),
        version_detection_method="explicit_document_id",
        matched_existing_document=True,
    )

    restored = prepared_document_from_payload(prepared_document_to_payload(prepared))

    assert restored == prepared


def test_background_job_routes_expose_status_and_evaluation_submission() -> None:
    queued = _job(job_type=BackgroundJobType.RUN_EVALUATION, payload={"dataset_path": "baseline.json"})

    class FakeGetHandler:
        async def __call__(self, query):
            assert query.job_id == queued.id
            return queued

    class FakeListHandler:
        async def __call__(self, query):
            assert query.limit == 10
            assert query.status is BackgroundJobStatus.QUEUED
            return [queued]

    class FakeEvaluationHandler:
        async def __call__(self, command):
            assert command.dataset_path == "baseline.json"
            assert command.repetitions == 2
            return queued

    app = create_app()
    app.dependency_overrides[get_background_job_handler] = lambda: FakeGetHandler()
    app.dependency_overrides[get_list_background_jobs_handler] = lambda: FakeListHandler()
    app.dependency_overrides[get_enqueue_evaluation_handler] = lambda: FakeEvaluationHandler()
    client = TestClient(app)

    listed = client.get("/api/v1/jobs", params={"limit": 10, "job_status": "queued"})
    fetched = client.get(f"/api/v1/jobs/{queued.id}")
    submitted = client.post(
        "/api/v1/jobs/evaluations",
        json={"dataset_path": "baseline.json", "repetitions": 2},
    )

    assert listed.status_code == 200
    assert listed.json()[0]["id"] == str(queued.id)
    assert fetched.status_code == 200
    assert fetched.json()["current_stage"] == "queued"
    assert submitted.status_code == 202
    assert submitted.json()["job_type"] == "run_evaluation"
    assert submitted.headers["location"].endswith(f"/api/v1/jobs/{queued.id}")


def test_upload_route_returns_processing_document_and_job_location() -> None:
    uow = FakeUnitOfWork()
    document = _run_async(uow.documents.get(uow.documents.document_id))
    assert document is not None
    queued = _job(payload={"document_id": str(document.id)})

    class FakeSubmitHandler:
        async def __call__(self, command):
            assert command.upload.filename == "architecture.md"
            return QueuedDocumentIngestionResult(document=document, job=queued)

    app = create_app()
    app.dependency_overrides[get_submit_document_ingestion_handler] = lambda: FakeSubmitHandler()
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("architecture.md", b"# Architecture", "text/markdown")},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "processing"
    assert response.headers["x-background-job-id"] == str(queued.id)
    assert response.headers["location"].endswith(f"/api/v1/jobs/{queued.id}")



def test_delete_document_version_route_returns_background_job_location() -> None:
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    queued = _job(
        job_type=BackgroundJobType.DELETE_DOCUMENT_VERSIONS,
        payload={
            "targets": [
                {
                    "document_id": str(document_id),
                    "document_version_id": str(version_id),
                }
            ]
        },
    )

    class FakeDeleteHandler:
        async def __call__(self, command):
            assert command.targets == (
                DocumentVersionDeletionTarget(
                    document_id=document_id,
                    version_id=version_id,
                ),
            )
            return queued

    app = create_app()
    app.dependency_overrides[get_enqueue_document_version_deletion_handler] = (
        lambda: FakeDeleteHandler()
    )
    client = TestClient(app)

    response = client.delete(
        f"/api/v1/documents/{document_id}/versions/{version_id}"
    )

    assert response.status_code == 202
    assert response.json() == {
        "job_id": str(queued.id),
        "status": "queued",
        "target_count": 1,
    }
    assert response.headers["x-background-job-id"] == str(queued.id)
    assert response.headers["location"].endswith(f"/api/v1/jobs/{queued.id}")


def test_batch_upload_returns_independent_jobs_and_rejections() -> None:
    uow = FakeUnitOfWork()
    document = _run_async(uow.documents.get(uow.documents.document_id))
    assert document is not None
    queued = _job(payload={"document_id": str(document.id)})

    class FakeSubmitHandler:
        async def __call__(self, command):
            if command.upload.filename == "bad.exe":
                raise UnsupportedDocumentTypeError("Unsupported document type.")
            return QueuedDocumentIngestionResult(document=document, job=queued)

    app = create_app()
    app.dependency_overrides[get_submit_document_ingestion_handler] = lambda: FakeSubmitHandler()
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents/batch",
        files=[
            ("files", ("architecture.md", b"# Architecture", "text/markdown")),
            ("files", ("bad.exe", b"bad", "application/octet-stream")),
        ],
    )

    assert response.status_code == 202
    body = response.json()
    assert [item["filename"] for item in body["accepted"]] == ["architecture.md"]
    assert body["accepted"][0]["job_id"] == str(queued.id)
    assert body["rejected"] == [
        {"filename": "bad.exe", "detail": "Unsupported document type."}
    ]


def _run_async(coroutine):
    import asyncio

    return asyncio.run(coroutine)


def test_worker_settings_reject_heartbeat_not_shorter_than_lease() -> None:
    try:
        Settings(
            _env_file=None,
            background_worker_heartbeat_seconds=60,
            background_worker_lock_timeout_seconds=60,
        )
    except ValueError as exc:
        assert "heartbeat interval" in str(exc)
    else:
        raise AssertionError("Expected invalid worker lease settings to be rejected.")


def test_chunk_index_and_point_ids_are_stable_for_retries() -> None:
    version_id = uuid.uuid4()

    assert chunk_index_id(version_id, 7) == chunk_index_id(version_id, 7)
    assert chunk_point_id(version_id, 7) == chunk_point_id(version_id, 7)
    assert chunk_index_id(version_id, 7) != chunk_index_id(version_id, 8)
    assert chunk_point_id(version_id, 7) != chunk_point_id(version_id, 8)

async def test_background_job_claim_refreshes_server_generated_fields_before_mapping() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from packages.indexer_infrastructure.postgres.models.background_jobs import BackgroundJob
    from packages.indexer_infrastructure.postgres.repositories.background_jobs import (
        SqlAlchemyBackgroundJobRepository,
    )

    refreshed_at = datetime(2026, 7, 31, 10, 0, tzinfo=UTC)
    model = BackgroundJob(
        id=uuid.uuid4(),
        job_type=BackgroundJobType.INGEST_DOCUMENT,
        status=BackgroundJobStatus.QUEUED,
        priority=100,
        payload={"document_id": str(uuid.uuid4())},
        result={},
        progress=0.0,
        current_stage="queued",
        attempts=0,
        max_attempts=3,
        scheduled_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )

    no_abandoned_job = MagicMock()
    no_abandoned_job.scalar_one_or_none.return_value = None
    queued_job = MagicMock()
    queued_job.scalar_one_or_none.return_value = model

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[no_abandoned_job, queued_job])
    session.flush = AsyncMock()

    async def refresh(instance: BackgroundJob) -> None:
        instance.updated_at = refreshed_at

    session.refresh = AsyncMock(side_effect=refresh)
    repository = SqlAlchemyBackgroundJobRepository(session)

    record = await repository.claim_next(
        worker_id="worker:test",
        now=refreshed_at,
        stale_before=NOW,
    )

    assert record is not None
    assert record.status is BackgroundJobStatus.RUNNING
    assert record.updated_at == refreshed_at
    session.flush.assert_awaited_once()
    session.refresh.assert_awaited_once_with(model)


def test_batch_delete_document_versions_route_preserves_explicit_targets() -> None:
    first_document_id = uuid.uuid4()
    second_document_id = uuid.uuid4()
    first_version_id = uuid.uuid4()
    second_version_id = uuid.uuid4()
    queued = _job(
        job_type=BackgroundJobType.DELETE_DOCUMENT_VERSIONS,
        payload={"targets": []},
    )

    class FakeDeleteHandler:
        async def __call__(self, command):
            assert command.targets == (
                DocumentVersionDeletionTarget(
                    document_id=first_document_id,
                    version_id=first_version_id,
                ),
                DocumentVersionDeletionTarget(
                    document_id=second_document_id,
                    version_id=second_version_id,
                ),
            )
            return queued

    app = create_app()
    app.dependency_overrides[get_enqueue_document_version_deletion_handler] = (
        lambda: FakeDeleteHandler()
    )
    client = TestClient(app)

    response = client.post(
        "/api/v1/documents/versions/batch-delete",
        json={
            "targets": [
                {
                    "document_id": str(first_document_id),
                    "document_version_id": str(first_version_id),
                },
                {
                    "document_id": str(second_document_id),
                    "document_version_id": str(second_version_id),
                },
            ]
        },
    )

    assert response.status_code == 202
    assert response.json()["target_count"] == 2
    assert response.json()["job_id"] == str(queued.id)
