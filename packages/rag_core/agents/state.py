from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposition
from packages.rag_core.query_understanding.planning import RetrievalPlan
from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem


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
    """Shared mutable state passed between agent graph nodes.

    This is the boundary object for query execution. Future agentic nodes can
    add classification, planning, grading, tool-use, and retry fields here
    without coupling API routes to individual retrieval or LLM implementations.
    """

    question: str
    top_k: int = 5
    query_run_id: uuid.UUID | None = None
    requested_pipeline_name: str | None = None
    pipeline_name: str | None = None
    pipeline_version: str | None = None
    query_classification: QueryClassification | None = None
    information_need_decomposition: InformationNeedDecomposition | None = None
    retrieval_plan: RetrievalPlan | None = None
    evidence_grading: EvidenceGradingReport | None = None
    retrieved_evidence: list[EvidenceItem] = field(default_factory=list)
    citations: list[CitationItem] = field(default_factory=list)
    answer: str | None = None
    trace: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
