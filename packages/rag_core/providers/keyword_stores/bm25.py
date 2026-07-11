from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

from packages.rag_core.providers.keyword_stores.base import (
    KeywordCorpusSource,
    KeywordDocument,
    KeywordSearchResult,
)

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class _IndexedDocument:
    document: KeywordDocument
    term_frequencies: Counter[str]
    length: int


@dataclass(frozen=True, slots=True)
class _BM25Index:
    documents: tuple[_IndexedDocument, ...]
    inverse_document_frequencies: dict[str, float]
    average_document_length: float


class BM25KeywordStore:
    """Small dependency-free BM25 keyword index built from an async corpus source.

    The corpus is cached for a bounded period so repeated evaluation cases do
    not repeatedly scroll the backing store. API ingestion clears the provider
    factory cache, while the TTL bounds staleness for external index changes.
    """

    def __init__(
        self,
        *,
        corpus_source: KeywordCorpusSource,
        k1: float = 1.5,
        b: float = 0.75,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative.")

        self._corpus_source = corpus_source
        self._k1 = k1
        self._b = b
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cached_index: _BM25Index | None = None
        self._cached_at = 0.0
        self._cache_lock = asyncio.Lock()

    async def search(self, query: str, *, top_k: int) -> list[KeywordSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        index = await self._get_index()
        if not index.documents:
            return []

        query_term_frequencies = Counter(query_terms)
        scored: list[tuple[float, KeywordDocument]] = []
        for indexed_document in index.documents:
            score = self._score_document(
                indexed_document=indexed_document,
                query_term_frequencies=query_term_frequencies,
                index=index,
            )
            if score > 0:
                scored.append((score, indexed_document.document))

        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            KeywordSearchResult(
                id=document.id,
                score=score,
                payload={"text": document.text, **document.payload},
            )
            for score, document in scored[:top_k]
        ]

    def invalidate(self) -> None:
        """Drop the cached corpus so the next search rebuilds the index."""

        self._cached_index = None
        self._cached_at = 0.0

    async def _get_index(self) -> _BM25Index:
        now = time.monotonic()
        if self._cache_is_fresh(now):
            return self._cached_index or _empty_index()

        async with self._cache_lock:
            now = time.monotonic()
            if self._cache_is_fresh(now):
                return self._cached_index or _empty_index()

            documents = await self._corpus_source.list_documents()
            self._cached_index = _build_index(documents)
            self._cached_at = now
            return self._cached_index

    def _cache_is_fresh(self, now: float) -> bool:
        if self._cached_index is None:
            return False
        if self._cache_ttl_seconds == 0:
            return False
        return now - self._cached_at < self._cache_ttl_seconds

    def _score_document(
        self,
        *,
        indexed_document: _IndexedDocument,
        query_term_frequencies: Counter[str],
        index: _BM25Index,
    ) -> float:
        if indexed_document.length == 0:
            return 0.0

        length_normalization = 1 - self._b
        if index.average_document_length > 0:
            length_normalization += self._b * indexed_document.length / index.average_document_length

        score = 0.0
        for term, query_frequency in query_term_frequencies.items():
            term_frequency = indexed_document.term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            inverse_document_frequency = index.inverse_document_frequencies.get(term, 0.0)
            numerator = term_frequency * (self._k1 + 1)
            denominator = term_frequency + self._k1 * length_normalization
            score += query_frequency * inverse_document_frequency * numerator / denominator
        return score


def _build_index(documents: list[KeywordDocument]) -> _BM25Index:
    indexed_documents: list[_IndexedDocument] = []
    document_frequencies: Counter[str] = Counter()

    for document in documents:
        terms = _tokenize(document.text)
        term_frequencies = Counter(terms)
        indexed_documents.append(
            _IndexedDocument(
                document=document,
                term_frequencies=term_frequencies,
                length=len(terms),
            ),
        )
        document_frequencies.update(term_frequencies.keys())

    document_count = len(indexed_documents)
    average_document_length = (
        sum(document.length for document in indexed_documents) / document_count if document_count else 0.0
    )
    inverse_document_frequencies = {
        term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequencies.items()
    }
    return _BM25Index(
        documents=tuple(indexed_documents),
        inverse_document_frequencies=inverse_document_frequencies,
        average_document_length=average_document_length,
    )


def _empty_index() -> _BM25Index:
    return _BM25Index(documents=(), inverse_document_frequencies={}, average_document_length=0.0)


def _tokenize(value: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(value)]
