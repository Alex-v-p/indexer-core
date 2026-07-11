from __future__ import annotations

import uuid

from packages.rag_core.ports import VectorSearchResult
from packages.rag_core.retrieval.retrievers import VectorRetriever


class FakeEmbeddingProvider:
    vector_size = 3

    def __init__(self) -> None:
        self.texts: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def __init__(self, hits: list[VectorSearchResult]) -> None:
        self.hits = hits
        self.searches: list[tuple[list[float], int]] = []
        self.ensure_calls = 0

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1

    async def search_by_vector(self, vector: list[float], *, top_k: int) -> list[VectorSearchResult]:
        self.searches.append((vector, top_k))
        return self.hits[:top_k]


async def test_vector_retriever_embeds_query_and_returns_evidence() -> None:
    chunk_index_id = uuid.uuid4()
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    embeddings = FakeEmbeddingProvider()
    vector_store = FakeVectorStore(
        [
            VectorSearchResult(
                id="qdrant-point-1",
                score=0.89,
                payload={
                    "text": "Graph nodes pass shared state between retrieval and generation.",
                    "qdrant_chunk_index_id": str(chunk_index_id),
                    "document_id": str(document_id),
                    "document_version_id": str(version_id),
                    "source_page_start": 4,
                    "original_filename": "architecture.md",
                },
            ),
        ],
    )
    retriever = VectorRetriever(embedding_provider=embeddings, vector_store=vector_store)

    evidence = await retriever.retrieve("How does the graph work?", top_k=5)

    assert embeddings.texts == [["How does the graph work?"]]
    assert vector_store.ensure_calls == 1
    assert vector_store.searches == [([0.1, 0.2, 0.3], 5)]
    assert len(evidence) == 1
    assert evidence[0].rank == 1
    assert evidence[0].score == 0.89
    assert evidence[0].text == "Graph nodes pass shared state between retrieval and generation."
    assert evidence[0].qdrant_chunk_index_id == chunk_index_id
    assert evidence[0].document_id == document_id
    assert evidence[0].document_version_id == version_id
    assert evidence[0].metadata["qdrant_point_id"] == "qdrant-point-1"
    assert evidence[0].metadata["retrieval_source"] == "vector"
    assert evidence[0].metadata["source_page_start"] == 4


async def test_vector_retriever_preserves_rank_order() -> None:
    retriever = VectorRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(
            [
                VectorSearchResult(id="a", score=0.9, payload={"text": "first"}),
                VectorSearchResult(id="b", score=0.8, payload={"text": "second"}),
            ],
        ),
    )

    evidence = await retriever.retrieve("question", top_k=2)

    assert [item.rank for item in evidence] == [1, 2]
    assert [item.text for item in evidence] == ["first", "second"]
