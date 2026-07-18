from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.agents.work_items import (
    InformationNeedExecution,
    InformationNeedResolutionReport,
)
from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning import RetrievalPlan
from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.retry.models import RetrievalRetryReport


@dataclass(slots=True)
class CitationItem:
    """A citation derived from retrieved evidence."""

    citation_index: int
    evidence_rank: int | None = None
    label: str | None = None
    page_number: int | None = None
    quote: str | None = None
    qdrant_chunk_index_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceEvent:
    """Runtime trace emitted by a graph runner."""

    step_order: int
    name: str
    step_type: str
    status: str
    duration_ms: int | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryState:
    """Shared mutable state passed between top-level and information-need graphs."""

    question: str
    top_k: int = 5
    query_run_id: uuid.UUID | None = None
    requested_pipeline_name: str | None = None
    pipeline_name: str | None = None
    pipeline_version: str | None = None
    query_classification: QueryClassification | None = None
    information_need_decomposition: InformationNeedDecomposition | None = None

    # Compatibility fields used by the fixed pipelines and the retrieval executor.
    retrieval_plan: RetrievalPlan | None = None
    active_retrieval_plan: RetrievalPlan | None = None
    active_retrieval_query: str | None = None
    active_retrieval_top_k: int | None = None
    evidence_grading: EvidenceGradingReport | None = None
    retrieval_retry: RetrievalRetryReport | None = None

    # Hierarchical agent state. Each information need owns its complete lifecycle.
    information_need_executions: dict[str, InformationNeedExecution] = field(default_factory=dict)
    pending_information_need_ids: list[str] = field(default_factory=list)
    active_information_need_id: str | None = None
    information_need_resolution: InformationNeedResolutionReport | None = None
    total_information_need_retrieval_attempts: int = 0
    evidence_by_key: dict[str, EvidenceItem] = field(default_factory=dict)

    retrieved_evidence: list[EvidenceItem] = field(default_factory=list)
    citations: list[CitationItem] = field(default_factory=list)
    answer: str | None = None
    trace: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    @property
    def effective_retrieval_plan(self) -> RetrievalPlan | None:
        return self.active_retrieval_plan or self.retrieval_plan

    @property
    def effective_retrieval_query(self) -> str:
        return self.active_retrieval_query or self.question

    @property
    def effective_retrieval_top_k(self) -> int:
        return self.active_retrieval_top_k or self.top_k

    @property
    def active_information_need_execution(self) -> InformationNeedExecution | None:
        if self.active_information_need_id is None:
            return None
        return self.information_need_executions.get(self.active_information_need_id)
