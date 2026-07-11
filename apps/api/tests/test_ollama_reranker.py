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


class IncompleteThenRecoveringReranker(OllamaReranker):
    def __init__(self) -> None:
        super().__init__(
            base_url="http://ollama",
            model="rerank-test",
            batch_size=4,
            max_attempts=2,
        )
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
        if len(candidate_ids) > 1:
            return {candidate_ids[0]: 0.9}
        return {candidate_ids[0]: 0.5 - (candidate_ids[0] * 0.01)}


class AlwaysIncompleteReranker(OllamaReranker):
    async def _score_batch(
        self,
        *,
        client: httpx.AsyncClient,
        question: str,
        batch: Sequence[tuple[int, EvidenceItem]],
    ) -> dict[int, float]:
        del client, question, batch
        return {}


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
        "score_source": "ollama",
        "fallback_used": False,
    }
    assert evidence[1].rank == 2
    assert "rerank" not in evidence[1].metadata


async def test_ollama_reranker_retries_omitted_candidates_individually() -> None:
    reranker = IncompleteThenRecoveringReranker()
    evidence = [
        EvidenceItem(rank=index + 1, text=f"candidate {index}", score=0.8 - index * 0.1)
        for index in range(4)
    ]

    reranked = await reranker.rerank("Which candidate is relevant?", evidence, top_k=4)

    assert reranker.batches == [[0, 1, 2, 3], [1], [2], [3]]
    assert len(reranked) == 4
    assert all(item.metadata["rerank"]["fallback_used"] is False for item in reranked)


async def test_ollama_reranker_preserves_original_order_when_retries_remain_incomplete() -> None:
    reranker = AlwaysIncompleteReranker(
        base_url="http://ollama",
        model="rerank-test",
        batch_size=3,
        max_attempts=2,
        fallback_to_original_rank=True,
    )
    evidence = [
        EvidenceItem(rank=1, text="first", score=0.8),
        EvidenceItem(rank=2, text="second", score=0.7),
        EvidenceItem(rank=3, text="third", score=0.6),
    ]

    reranked = await reranker.rerank("question", evidence, top_k=2)

    assert [item.text for item in reranked] == ["first", "second"]
    assert all(item.metadata["rerank"]["fallback_used"] is True for item in reranked)
    assert all(item.metadata["rerank"]["score_source"] == "original_rank_fallback" for item in reranked)


async def test_ollama_reranker_can_be_configured_to_fail_after_incomplete_retries() -> None:
    reranker = AlwaysIncompleteReranker(
        base_url="http://ollama",
        model="rerank-test",
        max_attempts=2,
        fallback_to_original_rank=False,
    )

    with pytest.raises(RerankerError, match="after 2 attempt"):
        await reranker.rerank("question", [EvidenceItem(rank=1, text="candidate")], top_k=1)


async def test_ollama_reranker_sends_exact_structured_schema_and_maps_local_ids() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://ollama/api/generate"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={"response": '{"scores":[{"id":0,"score":0.75},{"id":1,"score":0.25}]}'},
            request=request,
        )

    reranker = OllamaReranker(base_url="http://ollama", model="rerank-test")
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        scores = await reranker._score_batch(
            client=client,
            question="What is relevant?",
            batch=[
                (16, EvidenceItem(rank=17, text="Relevant evidence.")),
                (17, EvidenceItem(rank=18, text="Weak evidence.")),
            ],
        )

    assert scores == {16: 0.75, 17: 0.25}
    assert requests[0]["model"] == "rerank-test"
    assert requests[0]["stream"] is False
    assert requests[0]["format"]["properties"]["scores"]["minItems"] == 2
    assert requests[0]["format"]["properties"]["scores"]["maxItems"] == 2
    assert '"id": 0' in str(requests[0]["prompt"])
    assert '"id": 16' not in str(requests[0]["prompt"])
    assert "Relevant evidence." in str(requests[0]["prompt"])


def test_parse_rerank_scores_requires_one_valid_score_per_candidate_by_default() -> None:
    scores = _parse_rerank_scores(
        '{"scores":[{"id":0,"score":0.25},{"id":1,"score":1.0}]}',
        expected_ids={0, 1},
    )

    assert scores == {0: 0.25, 1: 1.0}

    with pytest.raises(RerankerError, match="omitted candidate ids"):
        _parse_rerank_scores('{"scores":[{"id":0,"score":0.25}]}', expected_ids={0, 1})

    partial = _parse_rerank_scores(
        '{"scores":[{"id":0,"score":0.25}]}',
        expected_ids={0, 1},
        require_complete=False,
    )
    assert partial == {0: 0.25}

    with pytest.raises(RerankerError, match="between 0 and 1"):
        _parse_rerank_scores('{"scores":[{"id":0,"score":1.5}]}', expected_ids={0})
