from __future__ import annotations

import pytest

from packages.rag_core.ports import KeywordStoreError
from packages.indexer_infrastructure.qdrant.keyword_corpus import _parse_scroll_page, _to_keyword_document


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
