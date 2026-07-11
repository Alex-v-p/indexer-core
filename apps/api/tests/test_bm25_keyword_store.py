from __future__ import annotations

from packages.rag_core.providers.keyword_stores import BM25KeywordStore, KeywordDocument


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
