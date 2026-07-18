from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
from packages.rag_core.retrieval.retrievers.base import Retriever, retrieve_compatibly


@dataclass(slots=True)
class _FusionCandidate:
    item: EvidenceItem
    score: float = 0.0
    sources: dict[str, dict[str, float | int | None]] = field(default_factory=dict)


class HybridRetriever:
    """Fuse dense-vector and keyword rankings with weighted reciprocal-rank fusion."""

    def __init__(
        self,
        *,
        vector_retriever: Retriever,
        keyword_retriever: Retriever,
        candidate_multiplier: int = 4,
        max_candidates: int = 100,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")
        if vector_weight < 0 or keyword_weight < 0:
            raise ValueError("fusion weights cannot be negative.")
        if vector_weight == 0 and keyword_weight == 0:
            raise ValueError("at least one fusion weight must be positive.")

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._candidate_multiplier = candidate_multiplier
        self._max_candidates = max_candidates
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> list[EvidenceItem]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        candidate_k = max(top_k, min(top_k * self._candidate_multiplier, self._max_candidates))
        vector_hits, keyword_hits = await asyncio.gather(
            retrieve_compatibly(self._vector_retriever, question, top_k=candidate_k, constraints=constraints),
            retrieve_compatibly(self._keyword_retriever, question, top_k=candidate_k, constraints=constraints),
        )

        candidates: dict[str, _FusionCandidate] = {}
        self._add_ranked_results(
            candidates=candidates,
            results=vector_hits,
            source_name="vector",
            weight=self._vector_weight,
        )
        self._add_ranked_results(
            candidates=candidates,
            results=keyword_hits,
            source_name="keyword",
            weight=self._keyword_weight,
        )

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.score, _evidence_key(candidate.item)),
        )
        return [self._to_fused_evidence(candidate, rank=rank) for rank, candidate in enumerate(ordered[:top_k], 1)]

    def _add_ranked_results(
        self,
        *,
        candidates: dict[str, _FusionCandidate],
        results: list[EvidenceItem],
        source_name: str,
        weight: float,
    ) -> None:
        if weight == 0:
            return

        for fallback_rank, item in enumerate(results, start=1):
            source_rank = item.rank if item.rank > 0 else fallback_rank
            key = _evidence_key(item)
            candidate = candidates.get(key)
            if candidate is None:
                candidate = _FusionCandidate(item=_copy_evidence(item))
                candidates[key] = candidate
            else:
                _merge_evidence(candidate.item, item)

            candidate.score += weight / (self._rrf_k + source_rank)
            candidate.sources[source_name] = {
                "rank": source_rank,
                "score": item.score,
                "weight": weight,
            }

    def _to_fused_evidence(self, candidate: _FusionCandidate, *, rank: int) -> EvidenceItem:
        metadata = dict(candidate.item.metadata)
        metadata["retrieval_source"] = "hybrid"
        metadata["fusion"] = {
            "method": "weighted_reciprocal_rank_fusion",
            "score": candidate.score,
            "rrf_k": self._rrf_k,
            "sources": candidate.sources,
        }
        return EvidenceItem(
            rank=rank,
            text=candidate.item.text,
            score=candidate.score,
            qdrant_chunk_index_id=candidate.item.qdrant_chunk_index_id,
            document_id=candidate.item.document_id,
            document_version_id=candidate.item.document_version_id,
            metadata=metadata,
        )


def _copy_evidence(item: EvidenceItem) -> EvidenceItem:
    return EvidenceItem(
        rank=item.rank,
        text=item.text,
        score=item.score,
        qdrant_chunk_index_id=item.qdrant_chunk_index_id,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        metadata=dict(item.metadata),
    )


def _merge_evidence(target: EvidenceItem, incoming: EvidenceItem) -> None:
    if not target.text and incoming.text:
        target.text = incoming.text
    if target.qdrant_chunk_index_id is None:
        target.qdrant_chunk_index_id = incoming.qdrant_chunk_index_id
    if target.document_id is None:
        target.document_id = incoming.document_id
    if target.document_version_id is None:
        target.document_version_id = incoming.document_version_id
    for key, value in incoming.metadata.items():
        target.metadata.setdefault(key, value)


def _evidence_key(item: EvidenceItem) -> str:
    if item.qdrant_chunk_index_id is not None:
        return f"chunk:{item.qdrant_chunk_index_id}"

    qdrant_point_id = item.metadata.get("qdrant_point_id") or item.metadata.get("keyword_store_document_id")
    if qdrant_point_id:
        return f"point:{qdrant_point_id}"

    if item.document_version_id is not None and "ordinal" in item.metadata:
        return f"version:{item.document_version_id}:ordinal:{item.metadata['ordinal']}"

    digest = hashlib.sha256(item.text.strip().encode("utf-8")).hexdigest()
    return f"text:{digest}"
