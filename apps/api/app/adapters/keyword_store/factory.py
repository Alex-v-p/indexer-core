from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings
from packages.rag_core.providers.keyword_stores import BM25KeywordStore, KeywordStore, QdrantKeywordCorpusSource


def build_keyword_store(settings: Settings) -> KeywordStore:
    """Build the configured keyword store with a process-local corpus cache."""

    return _build_qdrant_bm25_keyword_store(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.qdrant_timeout_seconds,
        settings.keyword_scroll_batch_size,
        settings.keyword_bm25_k1,
        settings.keyword_bm25_b,
        settings.keyword_cache_ttl_seconds,
    )


@lru_cache(maxsize=8)
def _build_qdrant_bm25_keyword_store(
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_timeout_seconds: float,
    scroll_batch_size: int,
    k1: float,
    b: float,
    cache_ttl_seconds: float,
) -> BM25KeywordStore:
    source = QdrantKeywordCorpusSource(
        base_url=qdrant_url,
        collection_name=qdrant_collection,
        timeout_seconds=qdrant_timeout_seconds,
        scroll_batch_size=scroll_batch_size,
    )
    return BM25KeywordStore(
        corpus_source=source,
        k1=k1,
        b=b,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def clear_keyword_store_cache() -> None:
    """Invalidate process-local keyword providers after chunk ingestion."""

    _build_qdrant_bm25_keyword_store.cache_clear()
