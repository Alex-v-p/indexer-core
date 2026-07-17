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
class RetrievalPlan:
    """Typed retrieval decision produced before evidence collection begins."""

    strategy: RetrievalStrategy
    selected_pipeline_name: str
    rationale: str
    planner_name: str
    based_on_query_type: QueryType
    metadata_filter_hints: tuple[MetadataFilterHint, ...] = ()
    requires_reranking: bool = False

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
        }
