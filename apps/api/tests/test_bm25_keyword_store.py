from __future__ import annotations

from datetime import UTC, datetime

from packages.indexer_infrastructure.bm25 import BM25KeywordStore
from packages.rag_core.ports import KeywordDocument
from packages.rag_core.query_understanding.temporal import (
    DateRange,
    DocumentDateConstraint,
    DocumentDateField,
)


class StaticCorpusSource:
    def __init__(self, documents: list[KeywordDocument]) -> None:
        self.documents = documents
        self.calls = 0

    async def list_documents(self) -> list[KeywordDocument]:
        self.calls += 1
        return self.documents


async def test_bm25_keyword_store_ranks_rare_exact_terms_first() -> None:
    source = StaticCorpusSource(
        [
            KeywordDocument(id="a", text="The architecture uses graph nodes and shared state."),
            KeywordDocument(id="b", text="The zxqv-42 recovery token appears only in this operational runbook."),
            KeywordDocument(id="c", text="The user interface displays citations and trace steps."),
        ],
    )
    store = BM25KeywordStore(corpus_source=source, cache_ttl_seconds=60)

    results = await store.search("zxqv 42 recovery token", top_k=3)

    assert results[0].id == "b"
    assert results[0].score > 0
    assert results[0].payload["text"].startswith("The zxqv-42")


async def test_bm25_keyword_store_reuses_cached_corpus() -> None:
    source = StaticCorpusSource([KeywordDocument(id="a", text="hybrid retrieval")])
    store = BM25KeywordStore(corpus_source=source, cache_ttl_seconds=60)

    await store.search("hybrid", top_k=1)
    await store.search("retrieval", top_k=1)

    assert source.calls == 1


async def test_bm25_keyword_store_returns_empty_for_tokenless_query() -> None:
    source = StaticCorpusSource([KeywordDocument(id="a", text="hybrid retrieval")])
    store = BM25KeywordStore(corpus_source=source)

    assert await store.search("---", top_k=5) == []
    assert source.calls == 0


async def test_bm25_keyword_store_can_score_contextual_text_and_return_original_evidence() -> None:
    source = StaticCorpusSource(
        [
            KeywordDocument(
                id="contextual",
                text="This chunk concerns the Apollo migration rollback procedure.",
                payload={"text": "Run rollback.sh after the health check fails.", "contextualization_status": "ready"},
            ),
        ],
    )
    store = BM25KeywordStore(corpus_source=source)

    results = await store.search("Apollo migration", top_k=1)

    assert results[0].payload["text"] == "Run rollback.sh after the health check fails."
    assert results[0].payload["contextualization_status"] == "ready"


async def test_bm25_keyword_store_filters_by_publication_date() -> None:
    source = StaticCorpusSource(
        [
            KeywordDocument(
                id="older",
                text="retry policy",
                payload={"published_at_epoch": datetime(2024, 5, 10, tzinfo=UTC).timestamp()},
            ),
            KeywordDocument(
                id="newer",
                text="retry policy",
                payload={"published_at_epoch": datetime(2025, 5, 10, tzinfo=UTC).timestamp()},
            ),
            KeywordDocument(id="unknown", text="retry policy", payload={}),
        ],
    )
    store = BM25KeywordStore(corpus_source=source)
    constraint = DocumentDateConstraint(
        field=DocumentDateField.PUBLISHED_AT,
        date_range=DateRange(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        original_expression="2024",
        rationale="Test publication range.",
        detector_name="test",
    )

    results = await store.search("retry", top_k=5, date_constraints=(constraint,))

    assert [item.id for item in results] == ["older"]


async def test_bm25_generic_recorded_date_matches_publication_or_upload_but_excludes_unknown() -> None:
    source = StaticCorpusSource(
        [
            KeywordDocument(
                id="published",
                text="metadata constraints",
                payload={"published_at_epoch": datetime(2026, 5, 2, tzinfo=UTC).timestamp()},
            ),
            KeywordDocument(
                id="uploaded",
                text="metadata constraints",
                payload={"uploaded_at_epoch": datetime(2026, 5, 8, tzinfo=UTC).timestamp()},
            ),
            KeywordDocument(
                id="outside",
                text="metadata constraints",
                payload={"uploaded_at_epoch": datetime(2026, 6, 1, tzinfo=UTC).timestamp()},
            ),
            KeywordDocument(id="unknown", text="metadata constraints", payload={}),
        ],
    )
    store = BM25KeywordStore(corpus_source=source)
    constraint = DocumentDateConstraint(
        field=DocumentDateField.ANY_RECORDED_AT,
        date_range=DateRange(
            start=datetime(2026, 5, 1, tzinfo=UTC),
            end=datetime(2026, 6, 1, tzinfo=UTC),
        ),
        original_expression="May 2026",
        rationale="Test generic recorded-date range.",
        detector_name="test",
    )

    results = await store.search("metadata", top_k=10, date_constraints=(constraint,))

    assert {item.id for item in results} == {"published", "uploaded"}
