from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.rerankers import RerankerError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CandidateScore:
    value: float
    source: str
    fallback_used: bool


class OllamaReranker:
    """Pointwise relevance reranker backed by Ollama structured generation.

    Candidate chunks are scored in bounded batches on a 0..1 relevance scale.
    Incomplete or malformed batch responses are retried as smaller requests.
    When configured, candidates still missing after all attempts retain their
    original retrieval order instead of failing the complete graph run.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        batch_size: int = 8,
        max_chars_per_candidate: int = 4000,
        max_attempts: int = 2,
        fallback_to_original_rank: bool = True,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_chars_per_candidate <= 0:
            raise ValueError("max_chars_per_candidate must be positive.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.max_chars_per_candidate = max_chars_per_candidate
        self.max_attempts = max_attempts
        self.fallback_to_original_rank = fallback_to_original_rank

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
        scores: dict[int, _CandidateScore] = {}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for start in range(0, len(indexed_candidates), self.batch_size):
                batch = indexed_candidates[start : start + self.batch_size]
                batch_scores = await self._score_batch_with_recovery(
                    client=client,
                    question=question,
                    batch=batch,
                    candidate_count=len(indexed_candidates),
                )
                scores.update(batch_scores)

        missing_ids = [candidate_id for candidate_id, _ in indexed_candidates if candidate_id not in scores]
        if missing_ids:
            raise RerankerError(f"Ollama reranker did not produce scores for candidate ids {missing_ids}.")

        ordered = sorted(
            indexed_candidates,
            key=lambda candidate: (
                scores[candidate[0]].fallback_used,
                -scores[candidate[0]].value,
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

    async def _score_batch_with_recovery(
        self,
        *,
        client: httpx.AsyncClient,
        question: str,
        batch: Sequence[tuple[int, EvidenceItem]],
        candidate_count: int,
    ) -> dict[int, _CandidateScore]:
        scored: dict[int, _CandidateScore] = {}
        pending = list(batch)
        errors: list[str] = []

        for attempt in range(1, self.max_attempts + 1):
            if not pending:
                break

            # The first request keeps normal batching for performance. Recovery
            # requests score one candidate at a time, which is substantially
            # easier for small instruction models to satisfy consistently.
            request_batches = [pending] if attempt == 1 else [[candidate] for candidate in pending]
            for request_batch in request_batches:
                try:
                    returned_scores = await self._score_batch(
                        client=client,
                        question=question,
                        batch=request_batch,
                    )
                except RerankerError as exc:
                    errors.append(str(exc))
                    continue
                for candidate_id, value in returned_scores.items():
                    scored[candidate_id] = _CandidateScore(
                        value=value,
                        source="ollama",
                        fallback_used=False,
                    )

            pending = [candidate for candidate in pending if candidate[0] not in scored]

        if pending and not self.fallback_to_original_rank:
            missing_ids = [candidate_id for candidate_id, _ in pending]
            details = f" Last provider error: {errors[-1]}" if errors else ""
            raise RerankerError(
                f"Ollama reranker omitted candidate ids {missing_ids} after "
                f"{self.max_attempts} attempt(s).{details}",
            )

        if pending:
            missing_ids = [candidate_id for candidate_id, _ in pending]
            logger.warning(
                "Ollama reranker omitted candidate ids %s after %s attempt(s); preserving original order.",
                missing_ids,
                self.max_attempts,
            )
            for candidate_id, item in pending:
                scored[candidate_id] = _CandidateScore(
                    value=_original_rank_fallback_score(
                        item,
                        fallback_rank=candidate_id + 1,
                        candidate_count=candidate_count,
                    ),
                    source="original_rank_fallback",
                    fallback_used=True,
                )

        return scored

    async def _score_batch(
        self,
        *,
        client: httpx.AsyncClient,
        question: str,
        batch: Sequence[tuple[int, EvidenceItem]],
    ) -> dict[int, float]:
        # Use compact request-local ids. Models are generally more reliable with
        # ids 0..N-1 than with arbitrary global ids from later batches.
        local_batch = [(local_id, item) for local_id, (_, item) in enumerate(batch)]
        local_to_global = {
            local_id: candidate_id
            for local_id, (candidate_id, _) in enumerate(batch)
        }
        response_schema = _rerank_response_schema(len(local_batch))
        try:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": _build_rerank_prompt(
                        question=question,
                        batch=local_batch,
                        max_chars_per_candidate=self.max_chars_per_candidate,
                        response_schema=response_schema,
                    ),
                    "stream": False,
                    "format": response_schema,
                    "options": {
                        "temperature": 0,
                        "seed": 0,
                        "num_predict": max(256, len(local_batch) * 64),
                    },
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

        local_scores = _parse_rerank_scores(
            raw_response,
            expected_ids=set(local_to_global),
            require_complete=False,
        )
        return {
            local_to_global[local_id]: value
            for local_id, value in local_scores.items()
        }


def _rerank_response_schema(candidate_count: int) -> dict[str, Any]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": candidate_count - 1,
                        },
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


def _build_rerank_prompt(
    *,
    question: str,
    batch: Sequence[tuple[int, EvidenceItem]],
    max_chars_per_candidate: int,
    response_schema: dict[str, Any],
) -> str:
    candidates = [
        {
            "id": candidate_id,
            "text": item.text.strip()[:max_chars_per_candidate],
        }
        for candidate_id, item in batch
    ]
    candidates_json = json.dumps(candidates, ensure_ascii=False)
    schema_json = json.dumps(response_schema, ensure_ascii=False)
    return (
        "Score each candidate passage for how directly it helps answer the query. "
        "Use a score from 0 to 1, where 1 means directly and completely relevant, "
        "and 0 means unrelated. Judge only relevance, not writing quality. Return "
        "exactly one score for every candidate id. Do not omit, duplicate, or invent "
        "ids and do not add commentary. Your response must match the JSON schema.\n\n"
        f"JSON schema:\n{schema_json}\n\n"
        f"Query:\n{question.strip()}\n\n"
        f"Candidates:\n{candidates_json}"
    )


def _parse_rerank_scores(
    raw_response: str,
    *,
    expected_ids: set[int],
    require_complete: bool = True,
) -> dict[int, float]:
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

    if require_complete and set(scores) != expected_ids:
        missing_ids = sorted(expected_ids - set(scores))
        raise RerankerError(f"Ollama reranker omitted candidate ids {missing_ids}.")
    return scores


def _reranked_copy(
    item: EvidenceItem,
    *,
    rank: int,
    rerank_score: _CandidateScore,
    model: str,
) -> EvidenceItem:
    metadata = dict(item.metadata)
    metadata["rerank"] = {
        "provider": "ollama",
        "model": model,
        "original_rank": item.rank,
        "original_score": item.score,
        "score": rerank_score.value,
        "score_source": rerank_score.source,
        "fallback_used": rerank_score.fallback_used,
    }
    return EvidenceItem(
        rank=rank,
        text=item.text,
        score=rerank_score.value,
        qdrant_chunk_index_id=item.qdrant_chunk_index_id,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        metadata=metadata,
    )


def _original_rank_fallback_score(
    item: EvidenceItem,
    *,
    fallback_rank: int,
    candidate_count: int,
) -> float:
    rank = _stable_original_rank(item, fallback_rank=fallback_rank)
    bounded_rank = min(max(rank, 1), max(candidate_count, 1))
    return (candidate_count - bounded_rank + 1) / max(candidate_count, 1)


def _stable_original_rank(item: EvidenceItem, *, fallback_rank: int) -> int:
    return item.rank if item.rank > 0 else fallback_rank
