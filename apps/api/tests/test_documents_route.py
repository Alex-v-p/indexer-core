import uuid
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.api.routes.documents import to_document_detail_response
from app.composition import (
    build_chunk_contextualizer,
    build_document_ingestion_config,
    build_subject_classification_policy,
)
from app.core.config import Settings
from app.main import create_app
from packages.indexer_application.dto import (
    DocumentRecord,
    DocumentStatus,
    DocumentVersionRecord,
    DocumentVersionStatus,
)


def test_documents_route_is_registered() -> None:
    client = TestClient(create_app())

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/documents" in response.json()["paths"]
    paths = response.json()["paths"]
    assert "/api/v1/documents/{document_id}/versions" in paths
    assert "/api/v1/documents/batch" in paths
    assert "/api/v1/documents/{document_id}/versions/{version_id}" in paths
    assert "/api/v1/documents/versions/batch-delete" in paths
    assert "/api/v1/documents/{document_id}/subject-classification" in paths
    assert "/api/v1/documents/subject-classification/backfill" in paths
    assert "delete" not in paths["/api/v1/documents/{document_id}"]
    upload_schemas = [
        schema
        for name, schema in response.json()["components"]["schemas"].items()
        if name.startswith("Body_upload_document")
    ]
    assert upload_schemas
    assert all("subject_ids" in schema["properties"] for schema in upload_schemas)
    summary_schema = response.json()["components"]["schemas"]["DocumentSummaryResponse"]
    assert "subject_classification" in summary_schema["properties"]


def test_default_document_ingestion_composition_enables_contextualization(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXTUALIZATION_ENABLED", raising=False)
    monkeypatch.delenv("CONTEXTUALIZATION_FAIL_OPEN", raising=False)
    settings = Settings(_env_file=None)

    config = build_document_ingestion_config(settings)

    assert config.contextualization_enabled is True
    assert config.contextualization_fail_open is False
    assert build_chunk_contextualizer(settings) is not None


def test_subject_classification_policy_uses_typed_settings() -> None:
    settings = Settings(
        _env_file=None,
        subject_classification_high_threshold=0.9,
        subject_classification_medium_threshold=0.7,
        subject_classification_policy_version="policy/test",
    )

    policy = build_subject_classification_policy(settings)

    assert policy.high_threshold == 0.9
    assert policy.medium_threshold == 0.7
    assert policy.policy_version == "policy/test"


def test_document_response_marks_latest_ready_version_not_failed_upload() -> None:
    now = datetime.now(UTC)
    versions = tuple(
        DocumentVersionRecord(
            id=uuid.uuid4(),
            version_number=number,
            storage_uri=f"file://v{number}.md",
            content_type="text/markdown",
            checksum_sha256=str(number),
            parser_name="markdown",
            parser_version="1",
            status=status,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        for number, status in (
            (1, DocumentVersionStatus.READY),
            (2, DocumentVersionStatus.FAILED),
        )
    )
    document = DocumentRecord(
        id=uuid.uuid4(),
        title="Policy",
        original_filename="policy.md",
        content_type="text/markdown",
        storage_uri="file://v1.md",
        size_bytes=100,
        checksum_sha256="1",
        status=DocumentStatus.READY,
        metadata={},
        created_at=now,
        updated_at=now,
        versions=versions,
    )

    response = to_document_detail_response(document)

    assert [version.is_latest for version in response.versions] == [True, False]


def test_read_document_route_invokes_application_query_handler() -> None:
    from app.dependencies.application import get_document_handler

    now = datetime.now(UTC)
    document_id = uuid.uuid4()
    document = DocumentRecord(
        id=document_id,
        title="Policy",
        original_filename="policy.md",
        content_type="text/markdown",
        storage_uri="file://policy.md",
        size_bytes=100,
        checksum_sha256="abc",
        status=DocumentStatus.READY,
        metadata={"chunk_count": 2},
        created_at=now,
        updated_at=now,
    )

    class FakeGetDocumentHandler:
        async def __call__(self, query):
            assert query.document_id == document_id
            return document

    app = create_app()
    app.dependency_overrides[get_document_handler] = lambda: FakeGetDocumentHandler()
    client = TestClient(app)

    response = client.get(f"/api/v1/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["title"] == "Policy"
    assert response.json()["chunk_count"] == 2
