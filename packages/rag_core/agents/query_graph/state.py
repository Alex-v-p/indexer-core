from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.agents.information_need_graph.models import InformationNeedExecution
from packages.rag_core.agents.information_need_graph.reporting import InformationNeedResolutionReport
from packages.rag_core.agents.runtime.models import TraceEvent
from packages.rag_core.generation.models import CitationItem
from packages.rag_core.documents import DocumentPreference
from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning import RetrievalPlan
from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.constraint_validation import ConstraintValidationReport
from packages.rag_core.retrieval.evidence_context import EvidenceContextBundle
from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.retry.models import RetrievalRetryReport


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

    # Compatibility fields used by fixed pipelines and the retrieval executor.
    retrieval_plan: RetrievalPlan | None = None
    active_retrieval_plan: RetrievalPlan | None = None
    active_retrieval_query: str | None = None
    active_retrieval_top_k: int | None = None
    evidence_grading: EvidenceGradingReport | None = None
    constraint_validation: ConstraintValidationReport | None = None
    evidence_context: EvidenceContextBundle | None = None
    retrieval_retry: RetrievalRetryReport | None = None

    # Hierarchical graph state. Each information need owns its complete lifecycle.
    information_need_executions: dict[str, InformationNeedExecution] = field(default_factory=dict)
    pending_information_need_ids: list[str] = field(default_factory=list)
    active_information_need_id: str | None = None
    information_need_resolution: InformationNeedResolutionReport | None = None
    total_information_need_retrieval_attempts: int = 0
    evidence_by_key: dict[str, EvidenceItem] = field(default_factory=dict)
    next_evidence_rank: int = 1
    primary_document_preference: DocumentPreference | None = None

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
