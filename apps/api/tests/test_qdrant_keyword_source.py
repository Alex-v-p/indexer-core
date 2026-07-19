from __future__ import annotations

import pytest

from packages.indexer_infrastructure.qdrant.keyword_corpus import _parse_scroll_page, _to_keyword_document
from packages.rag_core.ports import KeywordStoreError


def test_parse_qdrant_scroll_page() -> None:
    points, offset = _parse_scroll_page(
        {
            "result": {
                "points": [{"id": "point-1", "payload": {"text": "hello", "ordinal": 1}}],
                "next_page_offset": "point-1",
            },
        },
    )

    assert points[0]["id"] == "point-1"
    assert offset == "point-1"


def test_qdrant_point_becomes_keyword_document() -> None:
    document = _to_keyword_document(
        {"id": "point-1", "payload": {"text": "  searchable text  ", "ordinal": 1}},
    )

    assert document.id == "point-1"
    assert document.text == "searchable text"
    assert document.payload == {"ordinal": 1}


def test_parse_qdrant_scroll_page_rejects_invalid_shape() -> None:
    with pytest.raises(KeywordStoreError):
        _parse_scroll_page({"result": []})


def test_contextual_keyword_document_searches_context_but_returns_original_text() -> None:
    document = _to_keyword_document(
        {
            "id": "point-1",
            "payload": {
                "text": "Original source chunk.",
                "contextualized_text": "Document-aware context.\n\nOriginal source chunk.",
                "contextualization_status": "ready",
            },
        },
        search_text_field="contextualized_text",
        evidence_text_field="text",
        fallback_to_evidence_text=False,
    )

    assert document.text == "Document-aware context.\n\nOriginal source chunk."
    assert document.payload["text"] == "Original source chunk."
    assert "contextualized_text" not in document.payload
    assert document.payload["contextualization_status"] == "ready"


def test_contextual_keyword_document_does_not_fall_back_to_uncontextualized_text() -> None:
    document = _to_keyword_document(
        {
            "id": "point-1",
            "payload": {
                "text": "Original-only source chunk.",
                "contextualization_status": "not_indexed",
            },
        },
        search_text_field="contextualized_text",
        evidence_text_field="text",
        fallback_to_evidence_text=False,
    )

    assert document.text == ""
    assert document.payload["text"] == "Original-only source chunk."


class FakeScrollResponse:
    status_code = 200

    def json(self):
        return {"result": {"points": [], "next_page_offset": None}}

    def raise_for_status(self) -> None:
        return None


class FakeScrollClient:
    requests = []

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def post(self, url: str, *, json):
        self.requests.append((url, json))
        return FakeScrollResponse()


async def test_keyword_corpus_excludes_hierarchy_summary_points(monkeypatch: pytest.MonkeyPatch) -> None:
    from packages.indexer_infrastructure.qdrant.keyword_corpus import QdrantKeywordCorpusSource

    FakeScrollClient.requests = []
    monkeypatch.setattr(
        "packages.indexer_infrastructure.qdrant.keyword_corpus.httpx.AsyncClient",
        FakeScrollClient,
    )
    source = QdrantKeywordCorpusSource(
        base_url="http://qdrant.test",
        collection_name="chunks",
    )

    documents = await source.list_documents()

    assert documents == []
    assert FakeScrollClient.requests[0][1]["filter"] == {
        "must_not": [
            {
                "key": "point_type",
                "match": {"value": "hierarchy_summary"},
            },
        ],
    }
