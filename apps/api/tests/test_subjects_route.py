from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.dependencies.application import (
    get_create_subject_handler,
    get_document_subject_suggestion_review_handler,
    get_subject_name_resolution_handler,
    get_update_subject_handler,
)
from app.main import create_app
from packages.indexer_application.commands import (
    ArchivedSubjectMutationError,
    DocumentSubjectDecisionWriteConflict,
)
from packages.indexer_application.ports import SubjectCanonicalNameConflictError
from packages.indexer_application.dto import (
    DocumentSubjectDecisionRecord,
    SubjectNameMatchRecord,
    SubjectRecord,
)
from packages.rag_core.subjects import (
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
    SubjectKind,
    SubjectNameMatchType,
    InvalidSubjectNameError,
)


def _subject(name: str = "Orion") -> SubjectRecord:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    return SubjectRecord(
        id=uuid.uuid4(),
        kind=SubjectKind.PROJECT,
        name=name,
        normalized_name=name.casefold(),
        description=None,
        metadata={},
        created_at=now,
        updated_at=now,
    )


def _suggestion(*, revision: int = 3) -> DocumentSubjectDecisionRecord:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    return DocumentSubjectDecisionRecord(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.SUGGESTED,
        control_source=DecisionControlSource.AUTOMATIC,
        confidence=0.72,
        confidence_band=ConfidenceBand.MEDIUM,
        rationale="Classifier suggestion.",
        classifier_version="v1",
        policy_version="v1",
        signals={"title": True},
        classified_document_version_id=uuid.uuid4(),
        revision=revision,
        created_at=now,
        updated_at=now,
    )


def test_subject_administration_routes_and_contracts_are_registered() -> None:
    response = TestClient(create_app()).get("/openapi.json")

    assert response.status_code == 200
    body = response.json()
    paths = body["paths"]
    assert {"get", "post"}.issubset(paths["/api/v1/subjects"])
    assert {"get", "patch"}.issubset(paths["/api/v1/subjects/{subject_id}"])
    assert "get" in paths["/api/v1/subjects/resolve"]
    assert "post" in paths["/api/v1/subjects/{subject_id}/aliases"]
    assert "delete" in paths["/api/v1/subjects/{subject_id}/aliases/{alias_id}"]
    assert "get" in paths["/api/v1/documents/{document_id}/subjects"]
    assert "put" in paths["/api/v1/documents/{document_id}/subjects/{subject_id}"]
    assert "get" in paths["/api/v1/documents/{document_id}/subject-suggestions"]
    assert "post" in paths[
        "/api/v1/documents/{document_id}/subject-suggestions/{subject_id}/review"
    ]
    decision_request = body["components"]["schemas"][
        "SetDocumentSubjectDecisionRequest"
    ]
    assert set(decision_request["required"]) == {"state", "expected_revision"}
    for path, method in (
        ("/api/v1/documents/{document_id}/subjects/{subject_id}", "put"),
        (
            "/api/v1/documents/{document_id}/subject-suggestions/{subject_id}/review",
            "post",
        ),
    ):
        conflict_schema = paths[path][method]["responses"]["409"]["content"][
            "application/json"
        ]["schema"]
        assert conflict_schema["$ref"].endswith(
            "/SubjectDecisionConflictErrorResponse"
        )
    conflict_envelope = body["components"]["schemas"][
        "SubjectDecisionConflictErrorResponse"
    ]
    assert set(conflict_envelope["required"]) == {"detail"}


def test_legacy_document_response_contract_has_no_new_required_subject_fields() -> None:
    body = TestClient(create_app()).get("/openapi.json").json()
    summary = body["components"]["schemas"]["DocumentSummaryResponse"]
    detail = body["components"]["schemas"]["DocumentDetailResponse"]

    assert "subjects" not in summary["properties"]
    assert "subjects" not in detail.get("properties", {})
    assert "subject_decisions" not in summary["properties"]
    assert "subject_decisions" not in detail.get("properties", {})


def test_create_subject_route_maps_typed_request_and_response() -> None:
    subject = _subject()

    class FakeHandler:
        async def __call__(self, command):
            assert command.kind is SubjectKind.PROJECT
            assert command.name == "Orion"
            return subject

    app = create_app()
    app.dependency_overrides[get_create_subject_handler] = lambda: FakeHandler()
    response = TestClient(app).post(
        "/api/v1/subjects",
        json={"kind": "project", "name": "Orion"},
    )

    assert response.status_code == 201
    assert response.json()["id"] == str(subject.id)
    assert response.json()["kind"] == "project"


def test_invalid_normalized_subject_name_returns_422() -> None:
    class FakeHandler:
        async def __call__(self, command):
            raise InvalidSubjectNameError(
                "subject name must contain letters or numbers."
            )

    app = create_app()
    app.dependency_overrides[get_create_subject_handler] = lambda: FakeHandler()
    response = TestClient(app).post(
        "/api/v1/subjects",
        json={"kind": "project", "name": "---"},
    )

    assert response.status_code == 422
    assert "letters or numbers" in response.json()["detail"]


def test_canonical_name_race_returns_409() -> None:
    class FakeHandler:
        async def __call__(self, command):
            raise SubjectCanonicalNameConflictError(
                "A project subject with that canonical name already exists."
            )

    app = create_app()
    app.dependency_overrides[get_create_subject_handler] = lambda: FakeHandler()
    response = TestClient(app).post(
        "/api/v1/subjects",
        json={"kind": "project", "name": "Orion"},
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_archived_subject_mutation_returns_409() -> None:
    subject = _subject()

    class FakeHandler:
        async def __call__(self, command):
            raise ArchivedSubjectMutationError("Archived subjects are read-only.")

    app = create_app()
    app.dependency_overrides[get_update_subject_handler] = lambda: FakeHandler()
    response = TestClient(app).patch(
        f"/api/v1/subjects/{subject.id}",
        json={"name": "Renamed"},
    )

    assert response.status_code == 409
    assert "read-only" in response.json()["detail"]


def test_name_resolution_returns_all_colliding_aliases_as_ambiguous() -> None:
    matches = [
        SubjectNameMatchRecord(
            subject=_subject("Orion"),
            match_type=SubjectNameMatchType.ALIAS,
        ),
        SubjectNameMatchRecord(
            subject=_subject("Apollo"),
            match_type=SubjectNameMatchType.ALIAS,
        ),
    ]

    class FakeHandler:
        async def __call__(self, query):
            assert query.name == "shared"
            return matches

    app = create_app()
    app.dependency_overrides[get_subject_name_resolution_handler] = lambda: FakeHandler()
    response = TestClient(app).get("/api/v1/subjects/resolve?name=shared")

    assert response.status_code == 200
    assert response.json()["ambiguous"] is True
    assert len(response.json()["matches"]) == 2


def test_suggestion_revision_conflict_returns_current_context() -> None:
    current = _suggestion(revision=4)

    class FakeHandler:
        async def __call__(self, command):
            assert command.expected_revision == 3
            raise DocumentSubjectDecisionWriteConflict("Suggestion changed.", current=current)

    app = create_app()
    app.dependency_overrides[
        get_document_subject_suggestion_review_handler
    ] = lambda: FakeHandler()
    response = TestClient(app).post(
        f"/api/v1/documents/{current.document_id}/subject-suggestions/"
        f"{current.subject_id}/review",
        json={"decision": "accept", "expected_revision": 3},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["message"] == "Suggestion changed."
    assert detail["current"]["revision"] == 4
    assert detail["current"]["state"] == "suggested"
