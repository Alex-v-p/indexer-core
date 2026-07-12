from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.query_variants import QueryVariantGenerator
from packages.rag_core.retrieval.retrievers.base import RetrievalBatch, Retriever


@dataclass(frozen=True, slots=True)
class _QuerySpec:
    index: int
    text: str
    kind: str
    weight: float


@dataclass(slots=True)
class _FusionCandidate:
    item: EvidenceItem
    score: float = 0.0
    matches: list[dict[str, float | int | str | None]] = field(default_factory=list)


class MultiQueryRetriever:
    """Expand one question, retrieve each variant, and fuse rankings with RRF."""

    def __init__(
        self,
        *,
        query_variant_generator: QueryVariantGenerator,
        retriever: Retriever,
        base_retrieval_strategy: str = "configured_retriever",
        variant_count: int = 3,
        include_original: bool = True,
        candidate_multiplier: int = 2,
        max_candidates_per_query: int = 20,
        rrf_k: int = 60,
        original_query_weight: float = 1.2,
        variant_query_weight: float = 1.0,
        fail_open: bool = True,
    ) -> None:
        if not base_retrieval_strategy.strip():
            raise ValueError("base_retrieval_strategy must not be empty.")
        if variant_count <= 0:
            raise ValueError("variant_count must be positive.")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if max_candidates_per_query <= 0:
            raise ValueError("max_candidates_per_query must be positive.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")
        if original_query_weight < 0 or variant_query_weight < 0:
            raise ValueError("query weights cannot be negative.")
        if include_original and original_query_weight == 0 and variant_query_weight == 0:
            raise ValueError("at least one query weight must be positive.")
        if not include_original and variant_query_weight == 0:
            raise ValueError("variant_query_weight must be positive when the original query is excluded.")

        self._query_variant_generator = query_variant_generator
        self._retriever = retriever
        self._base_retrieval_strategy = base_retrieval_strategy.strip()
        self._variant_count = variant_count
        self._include_original = include_original
        self._candidate_multiplier = candidate_multiplier
        self._max_candidates_per_query = max_candidates_per_query
        self._rrf_k = rrf_k
        self._original_query_weight = original_query_weight
        self._variant_query_weight = variant_query_weight
        self._fail_open = fail_open

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        return (await self.retrieve_with_metadata(question, top_k=top_k)).evidence

    async def retrieve_with_metadata(self, question: str, *, top_k: int) -> RetrievalBatch:
        normalized_question = " ".join(question.strip().split())
        if not normalized_question:
            raise ValueError("question must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        generation_error: str | None = None
        generation_fallback_used = False
        try:
            variants = await self._query_variant_generator.generate(
                normalized_question,
                count=self._variant_count,
            )
        except Exception as exc:
            if not self._fail_open:
                raise
            variants = []
            generation_error = str(exc)
            generation_fallback_used = True

        query_specs = self._build_query_specs(normalized_question, variants)
        generated_variant_count = sum(query.kind == "variant" for query in query_specs)
        if generated_variant_count == 0:
            generation_fallback_used = True
        if not query_specs or not any(query.weight > 0 for query in query_specs):
            query_specs = [
                _QuerySpec(
                    index=0,
                    text=normalized_question,
                    kind="original_fallback",
                    weight=max(self._original_query_weight, self._variant_query_weight, 1.0),
                ),
            ]
            generation_fallback_used = True

        candidate_k = max(
            top_k,
            min(top_k * self._candidate_multiplier, self._max_candidates_per_query),
        )
        raw_results = await asyncio.gather(
            *(self._retriever.retrieve(query.text, top_k=candidate_k) for query in query_specs),
            return_exceptions=True,
        )

        successful_results: list[tuple[_QuerySpec, list[EvidenceItem]]] = []
        failed_queries: list[dict[str, str | int]] = []
        first_error: BaseException | None = None
        for query, result in zip(query_specs, raw_results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                if first_error is None:
                    first_error = result
                failed_queries.append(
                    {
                        "query_index": query.index,
                        "query": query.text,
                        "error": str(result),
                    },
                )
                continue
            successful_results.append((query, result))

        if first_error is not None and (not self._fail_open or not successful_results):
            raise first_error

        candidates: dict[str, _FusionCandidate] = {}
        for query, results in successful_results:
            self._add_ranked_results(
                candidates=candidates,
                query=query,
                results=results,
            )

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.score, _evidence_key(candidate.item)),
        )
        evidence = [
            self._to_fused_evidence(candidate, rank=rank)
            for rank, candidate in enumerate(ordered[:top_k], start=1)
        ]
        metadata: dict[str, Any] = {
            "strategy": "multi_query",
            "base_retrieval_strategy": self._base_retrieval_strategy,
            "fusion_method": "weighted_reciprocal_rank_fusion",
            "rrf_k": self._rrf_k,
            "requested_variant_count": self._variant_count,
            "generated_variant_count": generated_variant_count,
            "include_original": self._include_original,
            "candidate_top_k_per_query": candidate_k,
            "query_count": len(query_specs),
            "queries": [
                {
                    "index": query.index,
                    "text": query.text,
                    "kind": query.kind,
                    "weight": query.weight,
                }
                for query in query_specs
            ],
            "result_counts": {
                str(query.index): len(results)
                for query, results in successful_results
            },
            "failed_queries": failed_queries,
            "retrieval_fail_open_used": bool(failed_queries),
            "generation_fallback_used": generation_fallback_used,
        }
        if generation_error:
            metadata["generation_error"] = generation_error
        return RetrievalBatch(evidence=evidence, metadata=metadata)

    def _build_query_specs(self, original_question: str, variants: list[str]) -> list[_QuerySpec]:
        query_specs: list[_QuerySpec] = []
        seen: set[str] = set()

        if self._include_original:
            seen.add(_query_key(original_question))
            query_specs.append(
                _QuerySpec(
                    index=0,
                    text=original_question,
                    kind="original",
                    weight=self._original_query_weight,
                ),
            )

        for variant in variants:
            normalized = " ".join(variant.strip().split())
            key = _query_key(normalized)
            if not normalized or not key or key in seen:
                continue
            seen.add(key)
            query_specs.append(
                _QuerySpec(
                    index=len(query_specs),
                    text=normalized,
                    kind="variant",
                    weight=self._variant_query_weight,
                ),
            )
            if sum(query.kind == "variant" for query in query_specs) >= self._variant_count:
                break
        return query_specs

    def _add_ranked_results(
        self,
        *,
        candidates: dict[str, _FusionCandidate],
        query: _QuerySpec,
        results: list[EvidenceItem],
    ) -> None:
        if query.weight == 0:
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

            candidate.score += query.weight / (self._rrf_k + source_rank)
            candidate.matches.append(
                {
                    "query_index": query.index,
                    "query": query.text,
                    "query_kind": query.kind,
                    "rank": source_rank,
                    "score": item.score,
                    "weight": query.weight,
                },
            )

    def _to_fused_evidence(self, candidate: _FusionCandidate, *, rank: int) -> EvidenceItem:
        metadata = dict(candidate.item.metadata)
        metadata["retrieval_source"] = "multi_query"
        metadata["multi_query_fusion"] = {
            "method": "weighted_reciprocal_rank_fusion",
            "score": candidate.score,
            "rrf_k": self._rrf_k,
            "matches": candidate.matches,
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


def _query_key(value: str) -> str:
    return " ".join(value.casefold().split())
