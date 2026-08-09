from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Sequence

from packages.rag_core.document_organization.model_evidence import GroupContentCandidate


@dataclass(frozen=True, slots=True)
class SemanticGroupMatch:
    content_group_id: uuid.UUID
    score: float
    margin: float
    representative_content_hash: str
    document_count: int


def select_semantic_group_match(
    query_embedding: Sequence[float],
    candidates: tuple[GroupContentCandidate, ...],
    candidate_embeddings: Sequence[Sequence[float]],
    *,
    threshold: float,
    required_margin: float,
) -> SemanticGroupMatch | None:
    """Select a uniquely strong representative match without retaining vectors."""

    if not 0 <= threshold <= 1 or not 0 <= required_margin <= 1:
        raise ValueError("Semantic match threshold and margin must be between 0 and 1.")
    if len(candidates) != len(candidate_embeddings):
        raise ValueError("Semantic candidates and embeddings must have the same length.")
    if not candidates:
        return None

    scored = [
        (_cosine_similarity(query_embedding, embedding), str(candidate.content_group_id), candidate)
        for candidate, embedding in zip(candidates, candidate_embeddings, strict=True)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    top_score, _, top = scored[0]
    runner_score = scored[1][0] if len(scored) > 1 else 0.0
    observed_margin = top_score - runner_score
    if top_score < threshold or observed_margin < required_margin:
        return None
    return SemanticGroupMatch(
        content_group_id=top.content_group_id,
        score=top_score,
        margin=observed_margin,
        representative_content_hash=top.representative_content_hash,
        document_count=top.document_count,
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Embedding vectors must be non-empty and have the same dimensions.")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Embedding vectors must contain only finite values.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding vectors must have non-zero norms.")
    score = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))
