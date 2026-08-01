from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from packages.rag_core.ports import (
    EmbeddingProvider,
    VectorPayloadCondition,
    VectorSearchResult,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
from packages.rag_core.retrieval.retrievers.base import (
    RetrievalBatch,
    enforce_document_scope_with_count,
)
from packages.rag_core.retrieval.retrievers.vector import vector_result_to_evidence

_HIERARCHY_POINT_TYPE = "hierarchy_summary"
_DOCUMENT_LEVEL = "document"
_SECTION_LEVEL = "section"


class HierarchicalVectorStore(Protocol):
    """Vector-store capabilities required for multi-stage hierarchy traversal."""

    async def ensure_collection(self) -> None:
        """Ensure the collection exists and all configured vector names are available."""

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        vector_name: str,
        top_k: int,
        document_constraint=None,
        version_constraint=None,
        date_constraints=(),
        document_scope=None,
        payload_conditions: tuple[VectorPayloadCondition, ...] = (),
    ) -> list[VectorSearchResult]:
        """Search one named vector with metadata and hierarchy scope filters."""


@dataclass(frozen=True, slots=True)
class HierarchicalRetrieverConfig:
    """Candidate limits for document → section → chunk traversal."""

    document_candidates: int = 8
    section_candidates: int = 24
    chunk_candidate_multiplier: int = 4
    max_chunk_candidates: int = 80

    def __post_init__(self) -> None:
        if self.document_candidates <= 0:
            raise ValueError("document_candidates must be positive.")
        if self.section_candidates <= 0:
            raise ValueError("section_candidates must be positive.")
        if self.chunk_candidate_multiplier <= 0:
            raise ValueError("chunk_candidate_multiplier must be positive.")
        if self.max_chunk_candidates <= 0:
            raise ValueError("max_chunk_candidates must be positive.")


class HierarchicalRetriever:
    """Route broadly through document and section summaries before chunk search.

    Summary nodes select the most relevant document versions and semantic
    sections. The final search is then restricted to source chunks belonging to
    those selected sections, so only ordinary chunks become answer evidence.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: HierarchicalVectorStore,
        hierarchy_vector_name: str,
        chunk_vector_name: str,
        fallback_chunk_vector_name: str | None = None,
        config: HierarchicalRetrieverConfig | None = None,
    ) -> None:
        if not hierarchy_vector_name.strip():
            raise ValueError("hierarchy_vector_name must not be empty.")
        if not chunk_vector_name.strip():
            raise ValueError("chunk_vector_name must not be empty.")
        fallback = fallback_chunk_vector_name.strip() if fallback_chunk_vector_name else None
        if fallback == chunk_vector_name:
            fallback = None
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._hierarchy_vector_name = hierarchy_vector_name
        self._chunk_vector_name = chunk_vector_name
        self._fallback_chunk_vector_name = fallback
        self._config = config or HierarchicalRetrieverConfig()

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> list[EvidenceItem]:
        return (await self.retrieve_with_metadata(question, top_k=top_k, constraints=constraints)).evidence

    async def retrieve_with_metadata(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> RetrievalBatch:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if constraints is not None and constraints.document_scope.is_strict_empty:
            return RetrievalBatch(
                evidence=[],
                metadata=self._empty_metadata("strict_document_scope_empty"),
            )
        normalized_question = " ".join(question.strip().split())
        if not normalized_question:
            raise ValueError("question must not be empty.")

        embeddings = await self._embedding_provider.embed_texts([normalized_question])
        if not embeddings:
            return RetrievalBatch(evidence=[], metadata=self._empty_metadata("query_embedding_missing"))
        query_vector = embeddings[0]
        await self._vector_store.ensure_collection()

        document_hits = await self._search(
            query_vector,
            vector_name=self._hierarchy_vector_name,
            top_k=max(self._config.document_candidates, min(top_k, self._config.max_chunk_candidates)),
            constraints=constraints,
            payload_conditions=(
                VectorPayloadCondition("point_type", (_HIERARCHY_POINT_TYPE,)),
                VectorPayloadCondition("hierarchy_level", (_DOCUMENT_LEVEL,)),
            ),
        )
        selected_version_ids = _unique_payload_values(document_hits, "document_version_id")
        if not selected_version_ids:
            return RetrievalBatch(
                evidence=[],
                metadata=self._stage_metadata(
                    document_hits=document_hits,
                    section_hits=[],
                    chunk_hits=[],
                    selected_chunk_vector=self._chunk_vector_name,
                    stop_reason="no_document_summaries",
                ),
            )

        section_hits = await self._search(
            query_vector,
            vector_name=self._hierarchy_vector_name,
            top_k=max(self._config.section_candidates, top_k * 2),
            constraints=constraints,
            payload_conditions=(
                VectorPayloadCondition("point_type", (_HIERARCHY_POINT_TYPE,)),
                VectorPayloadCondition("hierarchy_level", (_SECTION_LEVEL,)),
                VectorPayloadCondition("document_version_id", tuple(selected_version_ids)),
            ),
        )
        selected_section_ids = _unique_payload_values(section_hits, "hierarchy_section_id")
        if not selected_section_ids:
            return RetrievalBatch(
                evidence=[],
                metadata=self._stage_metadata(
                    document_hits=document_hits,
                    section_hits=section_hits,
                    chunk_hits=[],
                    selected_chunk_vector=self._chunk_vector_name,
                    stop_reason="no_section_summaries",
                ),
            )

        chunk_candidate_count = min(
            self._config.max_chunk_candidates,
            max(top_k, top_k * self._config.chunk_candidate_multiplier),
        )
        chunk_conditions = (
            VectorPayloadCondition("retrieval_level", ("chunk",)),
            VectorPayloadCondition("hierarchy_section_id", tuple(selected_section_ids)),
        )
        selected_chunk_vector = self._chunk_vector_name
        chunk_hits = await self._search(
            query_vector,
            vector_name=selected_chunk_vector,
            top_k=chunk_candidate_count,
            constraints=constraints,
            payload_conditions=chunk_conditions,
        )
        fallback_used = False
        if not chunk_hits and self._fallback_chunk_vector_name is not None:
            selected_chunk_vector = self._fallback_chunk_vector_name
            chunk_hits = await self._search(
                query_vector,
                vector_name=selected_chunk_vector,
                top_k=chunk_candidate_count,
                constraints=constraints,
                payload_conditions=chunk_conditions,
            )
            fallback_used = True

        document_by_version = {
            str(hit.payload.get("document_version_id")): hit
            for hit in document_hits
            if hit.payload.get("document_version_id")
        }
        section_by_id = {
            str(hit.payload.get("hierarchy_section_id")): hit
            for hit in section_hits
            if hit.payload.get("hierarchy_section_id")
        }
        evidence: list[EvidenceItem] = []
        for rank, hit in enumerate(chunk_hits[:top_k], start=1):
            item = vector_result_to_evidence(rank=rank, hit=hit, vector_name=selected_chunk_vector)
            version_id = str(hit.payload.get("document_version_id") or "")
            section_id = str(hit.payload.get("hierarchy_section_id") or "")
            document_hit = document_by_version.get(version_id)
            section_hit = section_by_id.get(section_id)
            item.metadata["retrieval_source"] = "hierarchical"
            item.metadata["hierarchical_retrieval"] = {
                "document_summary": _summary_text(document_hit),
                "document_summary_score": document_hit.score if document_hit is not None else None,
                "section_summary": _summary_text(section_hit),
                "section_summary_score": section_hit.score if section_hit is not None else None,
                "hierarchy_section_id": section_id or None,
                "context_cluster_id": hit.payload.get("context_cluster_id"),
                "chunk_vector_name": selected_chunk_vector,
                "chunk_vector_fallback_used": fallback_used,
            }
            evidence.append(item)

        scoped_evidence, rejected = enforce_document_scope_with_count(
            evidence,
            constraints,
        )
        metadata = self._stage_metadata(
                document_hits=document_hits,
                section_hits=section_hits,
                chunk_hits=chunk_hits,
                selected_chunk_vector=selected_chunk_vector,
                fallback_used=fallback_used,
                stop_reason=None if evidence else "no_chunks_in_selected_sections",
            )
        metadata["out_of_scope_rejected_count"] = rejected
        return RetrievalBatch(
            evidence=scoped_evidence,
            metadata=metadata,
        )

    async def _search(
        self,
        vector: list[float],
        *,
        vector_name: str,
        top_k: int,
        constraints: RetrievalConstraints | None,
        payload_conditions: tuple[VectorPayloadCondition, ...],
    ) -> list[VectorSearchResult]:
        return await self._vector_store.search_by_vector(
            vector,
            vector_name=vector_name,
            top_k=top_k,
            document_constraint=constraints.document if constraints is not None else None,
            version_constraint=constraints.version if constraints is not None else None,
            date_constraints=constraints.dates if constraints is not None else (),
            document_scope=(
                constraints.document_scope
                if constraints is not None
                else RetrievalConstraints().document_scope
            ),
            payload_conditions=payload_conditions,
        )

    def _stage_metadata(
        self,
        *,
        document_hits: list[VectorSearchResult],
        section_hits: list[VectorSearchResult],
        chunk_hits: list[VectorSearchResult],
        selected_chunk_vector: str,
        fallback_used: bool = False,
        stop_reason: str | None,
    ) -> dict[str, object]:
        return {
            "strategy": "hierarchical_document_section_chunk",
            "hierarchy_vector_name": self._hierarchy_vector_name,
            "chunk_vector_name": selected_chunk_vector,
            "chunk_vector_fallback_used": fallback_used,
            "document_summary_candidate_count": len(document_hits),
            "section_summary_candidate_count": len(section_hits),
            "chunk_candidate_count": len(chunk_hits),
            "selected_document_version_ids": _unique_payload_values(document_hits, "document_version_id"),
            "selected_hierarchy_section_ids": _unique_payload_values(section_hits, "hierarchy_section_id"),
            "stop_reason": stop_reason,
        }

    def _empty_metadata(self, stop_reason: str) -> dict[str, object]:
        return self._stage_metadata(
            document_hits=[],
            section_hits=[],
            chunk_hits=[],
            selected_chunk_vector=self._chunk_vector_name,
            stop_reason=stop_reason,
        )


def _unique_payload_values(hits: list[VectorSearchResult], field: str) -> list[str]:
    return list(
        dict.fromkeys(
            str(value)
            for hit in hits
            if (value := hit.payload.get(field)) is not None and str(value)
        ),
    )


def _summary_text(hit: VectorSearchResult | None) -> str | None:
    if hit is None:
        return None
    value = hit.payload.get("summary_text") or hit.payload.get("text")
    return value.strip() if isinstance(value, str) and value.strip() else None
