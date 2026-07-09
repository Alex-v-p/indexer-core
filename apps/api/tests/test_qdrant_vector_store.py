from __future__ import annotations

from packages.rag_core.providers.vector_stores.qdrant import _parse_search_results


def test_parse_qdrant_search_endpoint_result() -> None:
    results = _parse_search_results(
        {
            "result": [
                {"id": "point-1", "score": 0.77, "payload": {"text": "hello"}},
            ],
        },
    )

    assert len(results) == 1
    assert results[0].id == "point-1"
    assert results[0].score == 0.77
    assert results[0].payload == {"text": "hello"}


def test_parse_qdrant_query_points_endpoint_result() -> None:
    results = _parse_search_results(
        {
            "result": {
                "points": [
                    {"id": "point-2", "score": 0.55, "payload": {"text": "world"}},
                ],
            },
        },
    )

    assert len(results) == 1
    assert results[0].id == "point-2"
    assert results[0].score == 0.55
    assert results[0].payload == {"text": "world"}
