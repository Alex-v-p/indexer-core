from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.rerankers import RerankerError

_RERANK_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 0},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["id", "score"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}


class OllamaReranker:
    """Pointwise relevance reranker backed by Ollama structured generation.

    Candidate chunks are scored in bounded batches on a 0..1 relevance scale.
    The implementation deliberately keeps the provider behind the core
    ``Reranker`` protocol so a dedicated cross-encoder or hosted reranking API
    can replace it without changing graph nodes or pipeline definitions.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        batch_size: int = 8,
        max_chars_per_candidate: int = 4000,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_chars_per_candidate <= 0:
            raise ValueError("max_chars_per_candidate must be positive.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.max_chars_per_candidate = max_chars_per_candidate

    async def rerank(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not question.strip():
            raise ValueError("question must not be empty.")
        if not evidence:
            return []

        indexed_candidates = list(enumerate(evidence))
        scores: dict[int, float] = {}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for start in range(0, len(indexed_candidates), self.batch_size):
                batch = indexed_candidates[start : start + self.batch_size]
                batch_scores = await self._score_batch(client=client, question=question, batch=batch)
                scores.update(batch_scores)

        missing_ids = [candidate_id for candidate_id, _ in indexed_candidates if candidate_id not in scores]
        if missing_ids:
            raise RerankerError(f"Ollama reranker did not return scores for candidate ids {missing_ids}.")

        ordered = sorted(
            indexed_candidates,
            key=lambda candidate: (
                -scores[candidate[0]],
                _stable_original_rank(candidate[1], fallback_rank=candidate[0] + 1),
                candidate[0],
            ),
        )
        selected = ordered[: min(top_k, len(ordered))]
        return [
            _reranked_copy(
                item,
                rank=rank,
                rerank_score=scores[candidate_id],
                model=self.model,
            )
            for rank, (candidate_id, item) in enumerate(selected, start=1)
        ]

    async def _score_batch(
        self,
        *,
        client: httpx.AsyncClient,
        question: str,
        batch: Sequence[tuple[int, EvidenceItem]],
    ) -> dict[int, float]:
        expected_ids = {candidate_id for candidate_id, _ in batch}
        try:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": _build_rerank_prompt(
                        question=question,
                        batch=batch,
                        max_chars_per_candidate=self.max_chars_per_candidate,
                    ),
                    "stream": False,
                    "format": _RERANK_RESPONSE_SCHEMA,
                    "options": {"temperature": 0, "seed": 0},
                },
            )
        except httpx.RequestError as exc:
            raise RerankerError(f"Ollama reranking request could not be completed: {exc}") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RerankerError(f"Ollama reranking request failed: {exc.response.text}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RerankerError("Ollama reranking response was not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise RerankerError("Ollama reranking response must be a JSON object.")

        raw_response = payload.get("response")
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise RerankerError("Ollama returned an empty reranking response.")
        return _parse_rerank_scores(raw_response, expected_ids=expected_ids)


def _build_rerank_prompt(
    *,
    question: str,
    batch: Sequence[tuple[int, EvidenceItem]],
    max_chars_per_candidate: int,
) -> str:
    candidates = [
        {
            "id": candidate_id,
            "text": item.text.strip()[:max_chars_per_candidate],
        }
        for candidate_id, item in batch
    ]
    candidates_json = json.dumps(candidates, ensure_ascii=False)
    return (
        "Score each candidate passage for how directly it helps answer the query. "
        "Use a score from 0 to 1, where 1 means directly and completely relevant, "
        "and 0 means unrelated. Judge only relevance, not writing quality. Return "
        "exactly one score for every candidate id and no extra commentary.\n\n"
        f"Query:\n{question.strip()}\n\n"
        f"Candidates:\n{candidates_json}"
    )


def _parse_rerank_scores(raw_response: str, *, expected_ids: set[int]) -> dict[int, float]:
    try:
        body = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RerankerError("Ollama reranker returned malformed structured output.") from exc

    if not isinstance(body, dict) or not isinstance(body.get("scores"), list):
        raise RerankerError("Ollama reranker response must contain a scores array.")

    scores: dict[int, float] = {}
    for entry in body["scores"]:
        if not isinstance(entry, dict):
            raise RerankerError("Ollama reranker returned a malformed score entry.")
        candidate_id = entry.get("id")
        score = entry.get("score")
        if not isinstance(candidate_id, int) or isinstance(candidate_id, bool):
            raise RerankerError("Ollama reranker score ids must be integers.")
        if candidate_id not in expected_ids:
            raise RerankerError(f"Ollama reranker returned unexpected candidate id {candidate_id}.")
        if candidate_id in scores:
            raise RerankerError(f"Ollama reranker returned duplicate candidate id {candidate_id}.")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise RerankerError(f"Ollama reranker score for candidate {candidate_id} must be numeric.")
        normalized_score = float(score)
        if not 0.0 <= normalized_score <= 1.0:
            raise RerankerError(f"Ollama reranker score for candidate {candidate_id} must be between 0 and 1.")
        scores[candidate_id] = normalized_score

    if set(scores) != expected_ids:
        missing_ids = sorted(expected_ids - set(scores))
        raise RerankerError(f"Ollama reranker omitted candidate ids {missing_ids}.")
    return scores


def _reranked_copy(item: EvidenceItem, *, rank: int, rerank_score: float, model: str) -> EvidenceItem:
    metadata = dict(item.metadata)
    metadata["rerank"] = {
        "provider": "ollama",
        "model": model,
        "original_rank": item.rank,
        "original_score": item.score,
        "score": rerank_score,
    }
    return EvidenceItem(
        rank=rank,
        text=item.text,
        score=rerank_score,
        qdrant_chunk_index_id=item.qdrant_chunk_index_id,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        metadata=metadata,
    )


def _stable_original_rank(item: EvidenceItem, *, fallback_rank: int) -> int:
    return item.rank if item.rank > 0 else fallback_rank
