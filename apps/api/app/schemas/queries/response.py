from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.jobs import BackgroundJobResponse

from .answer import AnswerPresentationResponse
from .retrieval import (
    ConstraintValidationResponse,
    DocumentPreferenceResponse,
    EvidenceContextResponse,
    EvidenceGradingResponse,
    InformationNeedDecompositionResponse,
    InformationNeedResolutionResponse,
    QueryClassificationResponse,
    RetrievalPlanResponse,
    RetrievalRetryResponse,
)
from .trace import TraceStepResponse


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


class QueryResponse(BaseModel):
    id: uuid.UUID
    question: str
    answer: str | None
    answer_presentation: AnswerPresentationResponse | None = None
    status: str
    pipeline_name: str | None
    pipeline_version: str | None
    top_k: int | None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    classification: QueryClassificationResponse | None = None
    information_need_decomposition: InformationNeedDecompositionResponse | None = None
    retrieval_plan: RetrievalPlanResponse | None = None
    evidence_grading: EvidenceGradingResponse | None = None
    retrieval_retry: RetrievalRetryResponse | None = None
    primary_document_preference: DocumentPreferenceResponse | None = None
    information_need_resolution: InformationNeedResolutionResponse | None = None
    constraint_validation: ConstraintValidationResponse | None = None
    evidence_context: EvidenceContextResponse | None = None
    evidence: list[EvidenceResponse] = Field(default_factory=list)
    citations: list[CitationResponse] = Field(default_factory=list)
    trace: list[TraceStepResponse] = Field(default_factory=list)


class QueuedQueryResponse(BaseModel):
    query: QueryResponse
    job: BackgroundJobResponse
