from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceItem:
    """A retrieved chunk snapshot used by the graph to generate an answer."""

    rank: int
    text: str
    score: float | None = None
    qdrant_chunk_index_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    """Runtime trace emitted by the graph runner."""

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
    """Shared mutable state passed between graph nodes.

    Keep this as the boundary object for query execution. Future agentic nodes
    can add planning/tool fields here without coupling API routes to individual
    retrievers, rerankers, or LLM providers.
    """

    question: str
    top_k: int = 5
    query_run_id: uuid.UUID | None = None
    pipeline_name: str = "baseline_rag"
    pipeline_version: str = "0.1.0"
    retrieved_evidence: list[EvidenceItem] = field(default_factory=list)
    citations: list[CitationItem] = field(default_factory=list)
    answer: str | None = None
    trace: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
