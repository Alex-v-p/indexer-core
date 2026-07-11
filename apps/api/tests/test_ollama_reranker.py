from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
import pytest

from packages.indexer_infrastructure.ollama.reranker import OllamaReranker, _parse_rerank_scores
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.rerankers import RerankerError


class StubOllamaReranker(OllamaReranker):
    def __init__(self, scores: dict[int, float]) -> None:
        super().__init__(base_url="http://ollama", model="rerank-test", batch_size=2)
        self.scores = scores
        self.batches: list[list[int]] = []

    async def _score_batch(
        self,
        *,
        client: httpx.AsyncClient,
        question: str,
        batch: Sequence[tuple[int, EvidenceItem]],
    ) -> dict[int, float]:
        del client, question
        candidate_ids = [candidate_id for candidate_id, _ in batch]
        self.batches.append(candidate_ids)
        return {candidate_id: self.scores[candidate_id] for candidate_id in candidate_ids}


async def test_ollama_reranker_batches_scores_and_preserves_original_retrieval_metadata() -> None:
    reranker = StubOllamaReranker({0: 0.2, 1: 0.95, 2: 0.6})
    evidence = [
        EvidenceItem(rank=1, text="weak", score=0.9, metadata={"retrieval_source": "hybrid"}),
        EvidenceItem(rank=2, text="best", score=0.5, metadata={"retrieval_source": "hybrid"}),
        EvidenceItem(rank=3, text="middle", score=0.7, metadata={"retrieval_source": "hybrid"}),
    ]

    reranked = await reranker.rerank("Which evidence is best?", evidence, top_k=2)

    assert reranker.batches == [[0, 1], [2]]
    assert [item.text for item in reranked] == ["best", "middle"]
    assert [item.rank for item in reranked] == [1, 2]
    assert [item.score for item in reranked] == [0.95, 0.6]
    assert reranked[0].metadata["retrieval_source"] == "hybrid"
    assert reranked[0].metadata["rerank"] == {
        "provider": "ollama",
        "model": "rerank-test",
        "original_rank": 2,
        "original_score": 0.5,
        "score": 0.95,
    }
    assert evidence[1].rank == 2
    assert "rerank" not in evidence[1].metadata


async def test_ollama_reranker_sends_structured_generate_request() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://ollama/api/generate"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={"response": '{"scores":[{"id":0,"score":0.75}]}'},
            request=request,
        )

    reranker = OllamaReranker(base_url="http://ollama", model="rerank-test")
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        scores = await reranker._score_batch(
            client=client,
            question="What is relevant?",
            batch=[(0, EvidenceItem(rank=1, text="Relevant evidence."))],
        )

    assert scores == {0: 0.75}
    assert requests[0]["model"] == "rerank-test"
    assert requests[0]["stream"] is False
    assert isinstance(requests[0]["format"], dict)
    assert "Relevant evidence." in str(requests[0]["prompt"])


def test_parse_rerank_scores_requires_one_valid_score_per_candidate() -> None:
    scores = _parse_rerank_scores(
        '{"scores":[{"id":0,"score":0.25},{"id":1,"score":1.0}]}',
        expected_ids={0, 1},
    )

    assert scores == {0: 0.25, 1: 1.0}

    with pytest.raises(RerankerError, match="omitted candidate ids"):
        _parse_rerank_scores('{"scores":[{"id":0,"score":0.25}]}', expected_ids={0, 1})

    with pytest.raises(RerankerError, match="between 0 and 1"):
        _parse_rerank_scores('{"scores":[{"id":0,"score":1.5}]}', expected_ids={0})
