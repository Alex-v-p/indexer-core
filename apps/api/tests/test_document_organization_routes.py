from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.dependencies.application import (
    get_create_document_type_handler,
    get_set_document_content_group_handler,
)
from app.main import create_app
from packages.indexer_application.commands import DocumentOrganizationWriteConflict
from packages.indexer_application.dto import DocumentContentGroupAssignmentRecord
from packages.indexer_application.ports import DocumentTypeKeyConflictError
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
)


def _assignment(document_id: uuid.UUID, group_id: uuid.UUID, revision: int = 3):
    now = datetime(2026, 8, 1, tzinfo=UTC)
    return DocumentContentGroupAssignmentRecord(
        document_id=document_id,
        content_group_id=group_id,
        state=ContentGroupAssignmentState.ASSIGNED,
        source=ClassificationSource.MANUAL,
        unresolved_reason=None,
        confidence=None,
        confidence_band=None,
        rationale="curated",
        classifier_version=None,
        policy_version=None,
        signals={},
        summary_hash=None,
        classified_document_version_id=None,
        revision=revision,
        created_at=now,
        updated_at=now,
    )


def test_document_organization_routes_and_cas_contracts_are_registered() -> None:
    response = TestClient(create_app()).get("/openapi.json")

    assert response.status_code == 200
    body = response.json()
    paths = body["paths"]
    assert {"get", "post"}.issubset(paths["/api/v1/document-types"])
    assert {"get", "patch"}.issubset(paths["/api/v1/document-types/{document_type_id}"])
    assert {"get", "post"}.issubset(paths["/api/v1/content-groups"])
    assert {"get", "patch"}.issubset(paths["/api/v1/content-groups/{content_group_id}"])
    assert "post" in paths["/api/v1/content-groups/{content_group_id}/aliases"]
    assert "delete" in paths[
        "/api/v1/content-groups/{content_group_id}/aliases/{alias_id}"
    ]
    assert "get" in paths["/api/v1/documents/{document_id}/organization"]
    assert "put" in paths["/api/v1/documents/{document_id}/organization/types"]
    assert "put" in paths[
        "/api/v1/documents/{document_id}/organization/content-group"
    ]
    assert "post" in paths["/api/v1/documents/{document_id}/organization/requeue"]
    assert "post" in paths["/api/v1/document-organization/backfill"]

    type_item = body["components"]["schemas"]["ManualDocumentTypeDecisionRequest"]
    assert {"document_type_id", "state", "expected_revision"}.issubset(type_item["required"])
    group_request = body["components"]["schemas"]["SetDocumentContentGroupRequest"]
    assert {"content_group_id", "expected_revision"}.issubset(group_request["required"])
    conflict = paths["/api/v1/documents/{document_id}/organization/types"]["put"][
        "responses"
    ]["409"]["content"]["application/json"]["schema"]
    assert conflict["$ref"].endswith("/DocumentOrganizationConflictErrorResponse")

    # The additive API must leave the legacy subject surface readable.
    assert "get" in paths["/api/v1/documents/{document_id}/subjects"]


def test_group_cas_conflict_is_409_with_current_revision() -> None:
    document_id = uuid.uuid4()
    group_id = uuid.uuid4()

    class FakeHandler:
        async def __call__(self, command):
            raise DocumentOrganizationWriteConflict(
                "stale group assignment",
                current_assignment=_assignment(document_id, group_id),
            )

    app = create_app()
    app.dependency_overrides[get_set_document_content_group_handler] = lambda: FakeHandler()
    response = TestClient(app).put(
        f"/api/v1/documents/{document_id}/organization/content-group",
        json={"content_group_id": str(group_id), "expected_revision": 2},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["current_assignment"]["revision"] == 3


def test_catalogue_conflict_is_409_and_request_validation_is_422() -> None:
    class ConflictHandler:
        async def __call__(self, command):
            raise DocumentTypeKeyConflictError("Document type key already exists.")

    app = create_app()
    app.dependency_overrides[get_create_document_type_handler] = lambda: ConflictHandler()
    client = TestClient(app)

    conflict = client.post(
        "/api/v1/document-types",
        json={"key": "report", "label": "Report"},
    )
    invalid = client.post(
        "/api/v1/document-types",
        json={"key": "", "label": ""},
    )

    assert conflict.status_code == 409
    assert invalid.status_code == 422


def test_manual_assignment_missing_document_is_404() -> None:
    class MissingHandler:
        async def __call__(self, command):
            raise LookupError(f"Document {command.document_id} was not found.")

    document_id = uuid.uuid4()
    app = create_app()
    app.dependency_overrides[get_set_document_content_group_handler] = lambda: MissingHandler()
    response = TestClient(app).put(
        f"/api/v1/documents/{document_id}/organization/content-group",
        json={"content_group_id": str(uuid.uuid4()), "expected_revision": 0},
    )

    assert response.status_code == 404
