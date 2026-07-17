from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from packages.rag_core.query_understanding.classification import MetadataFilterHint, QueryType


class RetrievalStrategy(StrEnum):
    """High-level retrieval behavior selected for a classified query."""

    BASELINE = "baseline"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    MULTI_QUERY = "multi_query"
    RERANK = "rerank"


@dataclass(frozen=True, slots=True)
class InformationNeed:
    """One atomic answer requirement extracted from the submitted question.

    These are pre-answer claims/aspects rather than generated factual claims.
    Evidence grading can therefore measure which parts of a compound question
    are covered before an answer is allowed to run.
    """

    need_id: str
    description: str
    retrieval_query: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.need_id.strip():
            raise ValueError("need_id must not be empty.")
        if not self.description.strip():
            raise ValueError("description must not be empty.")
        if not self.retrieval_query.strip():
            raise ValueError("retrieval_query must not be empty.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "need_id": self.need_id,
            "description": self.description,
            "retrieval_query": self.retrieval_query,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class InformationNeedDecomposition:
    """Structured decomposition produced as part of retrieval planning."""

    information_needs: tuple[InformationNeed, ...]
    rationale: str
    decomposer_name: str
    fallback_used: bool = False

    def __post_init__(self) -> None:
        if not self.information_needs:
            raise ValueError("At least one information need is required.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.decomposer_name.strip():
            raise ValueError("decomposer_name must not be empty.")
        ids = [need.need_id for need in self.information_needs]
        if len(ids) != len(set(ids)):
            raise ValueError("Information needs must have unique ids.")


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """Typed retrieval decision produced before evidence collection begins."""

    strategy: RetrievalStrategy
    selected_pipeline_name: str
    rationale: str
    planner_name: str
    based_on_query_type: QueryType
    metadata_filter_hints: tuple[MetadataFilterHint, ...] = ()
    requires_reranking: bool = False
    information_needs: tuple[InformationNeed, ...] = ()
    decomposition_rationale: str = ""
    decomposer_name: str = "none"
    decomposition_fallback_used: bool = False

    def __post_init__(self) -> None:
        if not self.selected_pipeline_name.strip():
            raise ValueError("selected_pipeline_name must not be empty.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.planner_name.strip():
            raise ValueError("planner_name must not be empty.")
        expected_reranking = self.strategy is RetrievalStrategy.RERANK
        if self.requires_reranking is not expected_reranking:
            raise ValueError("requires_reranking must match whether strategy is rerank.")
        ids = [need.need_id for need in self.information_needs]
        if len(ids) != len(set(ids)):
            raise ValueError("information_needs must have unique ids.")
        if self.information_needs and not self.decomposition_rationale.strip():
            raise ValueError("decomposition_rationale is required when information_needs are present.")
        if self.information_needs and not self.decomposer_name.strip():
            raise ValueError("decomposer_name is required when information_needs are present.")

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for persistence and tracing."""

        return {
            "strategy": self.strategy.value,
            "selected_pipeline_name": self.selected_pipeline_name,
            "rationale": self.rationale,
            "planner_name": self.planner_name,
            "based_on_query_type": self.based_on_query_type.value,
            "metadata_filter_hints": [hint.value for hint in self.metadata_filter_hints],
            "requires_reranking": self.requires_reranking,
            "information_needs": [need.to_metadata() for need in self.information_needs],
            "information_need_count": len(self.information_needs),
            "decomposition_rationale": self.decomposition_rationale,
            "decomposer_name": self.decomposer_name,
            "decomposition_fallback_used": self.decomposition_fallback_used,
        }
