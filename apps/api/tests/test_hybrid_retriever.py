from __future__ import annotations

import uuid

from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.retrievers import HybridRetriever


class StaticRetriever:
    def __init__(self, evidence: list[EvidenceItem]) -> None:
        self.evidence = evidence
        self.top_k_calls: list[int] = []

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        self.top_k_calls.append(top_k)
        return self.evidence[:top_k]


async def test_hybrid_retriever_deduplicates_and_rewards_cross_retriever_hits() -> None:
    shared_id = uuid.uuid4()
    vector = StaticRetriever(
        [
            EvidenceItem(rank=1, text="shared evidence", score=0.91, qdrant_chunk_index_id=shared_id),
            EvidenceItem(rank=2, text="vector only", score=0.82),
        ],
    )
    keyword = StaticRetriever(
        [
            EvidenceItem(rank=1, text="shared evidence", score=4.2, qdrant_chunk_index_id=shared_id),
            EvidenceItem(rank=2, text="keyword only", score=3.1),
        ],
    )
    retriever = HybridRetriever(
        vector_retriever=vector,
        keyword_retriever=keyword,
        candidate_multiplier=3,
        rrf_k=60,
    )

    evidence = await retriever.retrieve("question", top_k=2)

    assert vector.top_k_calls == [6]
    assert keyword.top_k_calls == [6]
    assert len(evidence) == 2
    assert evidence[0].qdrant_chunk_index_id == shared_id
    assert evidence[0].metadata["retrieval_source"] == "hybrid"
    assert set(evidence[0].metadata["fusion"]["sources"]) == {"vector", "keyword"}
    assert [item.rank for item in evidence] == [1, 2]


async def test_hybrid_retriever_can_surface_keyword_only_match() -> None:
    vector = StaticRetriever([EvidenceItem(rank=1, text="semantic match", score=0.88)])
    keyword = StaticRetriever([EvidenceItem(rank=1, text="ZXQ-491 exact code", score=6.4)])
    retriever = HybridRetriever(
        vector_retriever=vector,
        keyword_retriever=keyword,
        rrf_k=60,
        vector_weight=0.8,
        keyword_weight=1.2,
    )

    evidence = await retriever.retrieve("ZXQ-491", top_k=2)

    assert {item.text for item in evidence} == {"semantic match", "ZXQ-491 exact code"}
    keyword_hit = next(item for item in evidence if item.text.startswith("ZXQ"))
    assert "keyword" in keyword_hit.metadata["fusion"]["sources"]
