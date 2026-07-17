from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=25)
    pipeline_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_.-]*$",
        description="Registered pipeline to run. Omit to use the configured default.",
    )


class QueryClassificationResponse(BaseModel):
    query_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_metadata_filters: bool
    metadata_filter_hints: list[str] = Field(default_factory=list)
    rationale: str
    classifier_name: str
    fallback_used: bool = False


class EvidenceResponse(BaseModel):
    id: uuid.UUID | None = None
    rank: int
    score: float | None = None
    text: str
    qdrant_chunk_index_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationResponse(BaseModel):
    id: uuid.UUID | None = None
    citation_index: int
    label: str | None = None
    evidence_id: uuid.UUID | None = None
    page_number: int | None = None
    quote: str | None = None
    qdrant_chunk_index_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStepResponse(BaseModel):
    id: uuid.UUID | None = None
    step_order: int
    name: str
    step_type: str | None = None
    status: str
    duration_ms: int | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    id: uuid.UUID
    question: str
    answer: str | None
    status: str
    pipeline_name: str | None
    pipeline_version: str | None
    top_k: int | None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    classification: QueryClassificationResponse | None = None
    evidence: list[EvidenceResponse] = Field(default_factory=list)
    citations: list[CitationResponse] = Field(default_factory=list)
    trace: list[TraceStepResponse] = Field(default_factory=list)
