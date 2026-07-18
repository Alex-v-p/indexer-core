from __future__ import annotations

from dataclasses import dataclass

from packages.rag_core.query_understanding.planning import RetrievalStrategy
from packages.rag_core.retrieval.rerankers import Reranker
from packages.rag_core.retrieval.retrievers import Retriever


@dataclass(frozen=True, slots=True)
class RetrievalPlanExecution:
    """Runtime dependencies needed to execute one selectable retrieval pipeline."""

    pipeline_name: str
    pipeline_version: str
    strategy: RetrievalStrategy
    retriever: Retriever
    reranker: Reranker | None = None
    candidate_multiplier: int = 1
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.pipeline_name.strip():
            raise ValueError("pipeline_name must not be empty.")
        if not self.pipeline_version.strip():
            raise ValueError("pipeline_version must not be empty.")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")
        expected_reranker = self.strategy is RetrievalStrategy.RERANK
        if (self.reranker is not None) is not expected_reranker:
            raise ValueError("reranker configuration must match whether strategy is rerank.")
