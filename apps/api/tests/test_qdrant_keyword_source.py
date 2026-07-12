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
