from __future__ import annotations

import uuid

from packages.rag_core.providers.keyword_stores import KeywordSearchResult
from packages.rag_core.retrieval.retrievers import KeywordRetriever


class FakeKeywordStore:
    async def search(self, query: str, *, top_k: int) -> list[KeywordSearchResult]:
        return [
            KeywordSearchResult(
                id="point-1",
                score=3.7,
                payload={
                    "text": "BM25 finds exact identifiers that dense retrieval can miss.",
                    "qdrant_chunk_index_id": str(CHUNK_ID),
                    "document_id": str(DOCUMENT_ID),
                    "document_version_id": str(VERSION_ID),
                    "ordinal": 2,
                },
            ),
        ][:top_k]


CHUNK_ID = uuid.uuid4()
DOCUMENT_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()


async def test_keyword_retriever_normalizes_keyword_hits_to_evidence() -> None:
    retriever = KeywordRetriever(keyword_store=FakeKeywordStore())

    evidence = await retriever.retrieve("exact identifier", top_k=5)

    assert len(evidence) == 1
    assert evidence[0].rank == 1
    assert evidence[0].score == 3.7
    assert evidence[0].qdrant_chunk_index_id == CHUNK_ID
    assert evidence[0].document_id == DOCUMENT_ID
    assert evidence[0].document_version_id == VERSION_ID
    assert evidence[0].metadata["retrieval_source"] == "keyword"
    assert evidence[0].metadata["keyword_store_document_id"] == "point-1"
