from __future__ import annotations

from datetime import UTC, datetime

from typing import Any

import pytest

from packages.indexer_infrastructure.qdrant.vector_store import QdrantVectorStore
from packages.rag_core.documents import DocumentVersionConstraint, VersionSelectionMode
from packages.rag_core.query_understanding.temporal import (
    DateRange,
    DocumentDateConstraint,
    DocumentDateField,
)
from packages.rag_core.ports import VectorPoint, VectorStoreError


class FakeResponse:
    def __init__(self, *, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = __import__("httpx").Request("GET", "http://qdrant.test")
            response = __import__("httpx").Response(self.status_code, request=request, json=self._body)
            raise __import__("httpx").HTTPStatusError("request failed", request=request, response=response)


class FakeAsyncClient:
    responses: list[FakeResponse] = []
    requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def get(self, url: str) -> FakeResponse:
        self.requests.append(("GET", url, None))
        return self.responses.pop(0)

    async def put(self, url: str, *, json: dict[str, Any], params: dict[str, Any] | None = None) -> FakeResponse:
        body = {"json": json, "params": params}
        self.requests.append(("PUT", url, body))
        return self.responses.pop(0)

    async def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
        self.requests.append(("POST", url, {"json": json}))
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def reset_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncClient.responses = []
    FakeAsyncClient.requests = []
    monkeypatch.setattr(
        "packages.indexer_infrastructure.qdrant.vector_store.httpx.AsyncClient",
        FakeAsyncClient,
    )


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(
        base_url="http://qdrant.test",
        collection_name="chunks",
        vector_size=3,
        vector_names=("original", "contextual"),
    )


async def test_qdrant_collection_is_created_with_named_vectors() -> None:
    FakeAsyncClient.responses = [
        FakeResponse(status_code=404, body={}),
        FakeResponse(status_code=200, body={"result": True}),
    ]

    await _store().ensure_collection()

    create_request = FakeAsyncClient.requests[1]
    assert create_request[0] == "PUT"
    assert create_request[2]["json"]["vectors"] == {
        "original": {"size": 3, "distance": "Cosine"},
        "contextual": {"size": 3, "distance": "Cosine"},
    }


async def test_qdrant_upsert_writes_both_vectors_on_one_point() -> None:
    FakeAsyncClient.responses = [FakeResponse(status_code=200, body={"result": {}})]
    point = VectorPoint(
        id="point-1",
        vectors={
            "original": [0.1, 0.2, 0.3],
            "contextual": [0.4, 0.5, 0.6],
        },
        payload={"text": "Original", "contextualized_text": "Context\n\nOriginal"},
    )

    await _store().upsert_points([point])

    upsert_body = FakeAsyncClient.requests[0][2]["json"]
    assert upsert_body["points"] == [
        {
            "id": "point-1",
            "vector": point.vectors,
            "payload": point.payload,
        },
    ]


async def test_qdrant_query_selects_named_vector() -> None:
    FakeAsyncClient.responses = [
        FakeResponse(
            status_code=200,
            body={"result": {"points": [{"id": "point-1", "score": 0.9, "payload": {"text": "Original"}}]}},
        ),
    ]

    results = await _store().search_by_vector(
        [0.1, 0.2, 0.3],
        vector_name="contextual",
        top_k=4,
    )

    query = FakeAsyncClient.requests[0]
    assert query[1].endswith("/collections/chunks/points/query")
    assert query[2]["json"]["using"] == "contextual"
    assert query[2]["json"]["query"] == [0.1, 0.2, 0.3]
    assert results[0].id == "point-1"


async def test_qdrant_query_applies_explicit_latest_version_filter() -> None:
    FakeAsyncClient.responses = [FakeResponse(status_code=200, body={"result": {"points": []}})]

    await _store().search_by_vector(
        [0.1, 0.2, 0.3],
        vector_name="original",
        top_k=4,
        version_constraint=DocumentVersionConstraint(
            mode=VersionSelectionMode.LATEST,
            confidence=1.0,
            rationale="Test latest filter.",
            detector_name="test",
        ),
    )

    query_body = FakeAsyncClient.requests[0][2]["json"]
    assert query_body["filter"] == {
        "must": [{"key": "is_latest_version", "match": {"value": True}}],
    }


async def test_existing_unnamed_collection_is_rejected_with_clear_error() -> None:
    FakeAsyncClient.responses = [
        FakeResponse(
            status_code=200,
            body={
                "result": {
                    "config": {
                        "params": {
                            "vectors": {"size": 3, "distance": "Cosine"},
                        },
                    },
                },
            },
        ),
    ]

    with pytest.raises(VectorStoreError, match="unnamed vector"):
        await _store().ensure_collection()


async def test_qdrant_query_combines_version_and_publication_filters() -> None:
    FakeAsyncClient.responses = [FakeResponse(status_code=200, body={"result": {"points": []}})]
    date_constraint = DocumentDateConstraint(
        field=DocumentDateField.PUBLISHED_AT,
        date_range=DateRange(
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        original_expression="2025",
        rationale="Test publication range.",
        detector_name="test",
    )

    await _store().search_by_vector(
        [0.1, 0.2, 0.3],
        vector_name="original",
        top_k=4,
        version_constraint=DocumentVersionConstraint(
            mode=VersionSelectionMode.LATEST,
            confidence=1.0,
            rationale="Test latest filter.",
            detector_name="test",
        ),
        date_constraints=(date_constraint,),
    )

    query_body = FakeAsyncClient.requests[0][2]["json"]
    assert query_body["filter"]["must"] == [
        {
            "key": "published_at_epoch",
            "range": {
                "gte": datetime(2025, 1, 1, tzinfo=UTC).timestamp(),
                "lt": datetime(2026, 1, 1, tzinfo=UTC).timestamp(),
            },
        },
    ]


async def test_qdrant_query_filters_generic_recorded_date_against_publication_or_upload() -> None:
    FakeAsyncClient.responses = [FakeResponse(status_code=200, body={"result": {"points": []}})]
    date_constraint = DocumentDateConstraint(
        field=DocumentDateField.ANY_RECORDED_AT,
        date_range=DateRange(
            start=datetime(2026, 5, 1, tzinfo=UTC),
            end=datetime(2026, 6, 1, tzinfo=UTC),
        ),
        original_expression="May 2026",
        rationale="Test generic recorded-date range.",
        detector_name="test",
    )

    await _store().search_by_vector(
        [0.1, 0.2, 0.3],
        vector_name="original",
        top_k=4,
        date_constraints=(date_constraint,),
    )

    query_body = FakeAsyncClient.requests[0][2]["json"]
    expected_range = {
        "gte": datetime(2026, 5, 1, tzinfo=UTC).timestamp(),
        "lt": datetime(2026, 6, 1, tzinfo=UTC).timestamp(),
    }
    assert query_body["filter"]["must"] == [
        {
            "should": [
                {"key": "published_at_epoch", "range": expected_range},
                {"key": "uploaded_at_epoch", "range": expected_range},
            ],
        },
    ]
