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


class VersionConstraintResponse(BaseModel):
    mode: str = "all"
    version_numbers: list[int] = Field(default_factory=list)
    active: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = "No version-specific constraint was recorded."
    detector_name: str = "none"


class DocumentNameConstraintResponse(BaseModel):
    names: list[str] = Field(default_factory=list)
    normalized_names: list[str] = Field(default_factory=list)
    active: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = "No document-name constraint was recorded."
    detector_name: str = "none"
    match_semantics: str = "exact_normalized_any"


class DateRangeResponse(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    end_exclusive: bool = True


class DateConstraintResponse(BaseModel):
    field: str
    range: DateRangeResponse
    original_expression: str
    active: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str
    detector_name: str


class ConstraintValidationResponse(BaseModel):
    status: str
    blocked: bool = False
    candidate_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    constraints: dict[str, Any] = Field(default_factory=dict)
    rationale: str


class EvidenceSourceContextResponse(BaseModel):
    evidence_rank: int = Field(ge=1)
    values: dict[str, str] = Field(default_factory=dict)


class EvidenceContextResponse(BaseModel):
    constraint_summary: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    validation: ConstraintValidationResponse
    sources: list[EvidenceSourceContextResponse] = Field(default_factory=list)


class QueryClassificationResponse(BaseModel):
    query_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_metadata_filters: bool
    metadata_filter_hints: list[str] = Field(default_factory=list)
    rationale: str
    classifier_name: str
    fallback_used: bool = False
    document_constraint: DocumentNameConstraintResponse = Field(default_factory=DocumentNameConstraintResponse)
    version_constraint: VersionConstraintResponse = Field(default_factory=VersionConstraintResponse)
    date_constraints: list[DateConstraintResponse] = Field(default_factory=list)


class InformationNeedResponse(BaseModel):
    need_id: str
    description: str
    retrieval_query: str
    required: bool = True


class InformationNeedDecompositionResponse(BaseModel):
    information_needs: list[InformationNeedResponse] = Field(default_factory=list)
    information_need_count: int = Field(default=0, ge=0)
    rationale: str
    decomposer_name: str
    fallback_used: bool = False


class RetrievalPlanResponse(BaseModel):
    strategy: str
    selected_pipeline_name: str
    rationale: str
    planner_name: str
    based_on_query_type: str
    metadata_filter_hints: list[str] = Field(default_factory=list)
    requires_reranking: bool = False
    target_information_need_ids: list[str] = Field(default_factory=list)
    target_information_need_count: int = Field(default=0, ge=0)
    document_constraint: DocumentNameConstraintResponse = Field(default_factory=DocumentNameConstraintResponse)
    version_constraint: VersionConstraintResponse = Field(default_factory=VersionConstraintResponse)
    date_constraints: list[DateConstraintResponse] = Field(default_factory=list)


class EvidenceGradeResponse(BaseModel):
    evidence_rank: int = Field(ge=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
    relevant: bool
    rationale: str
    supports_information_need_ids: list[str] = Field(default_factory=list)


class InformationNeedGradeResponse(BaseModel):
    information_need_id: str
    description: str
    status: str
    coverage_score: float = Field(ge=0.0, le=1.0)
    supporting_evidence_ranks: list[int] = Field(default_factory=list)
    rationale: str
    required: bool = True


class EvidenceGradingResponse(BaseModel):
    status: str
    coverage_score: float = Field(ge=0.0, le=1.0)
    sufficient: bool
    answerable: bool = False
    partial_answer_available: bool = False
    missing_evidence: bool
    weak_evidence: bool
    relevant_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    relevant_evidence_ranks: list[int] = Field(default_factory=list)
    supported_information_need_count: int = Field(default=0, ge=0)
    supported_required_information_need_count: int = Field(default=0, ge=0)
    required_information_need_count: int = Field(default=0, ge=0)
    supported_information: list[str] = Field(default_factory=list)
    partial_information_need_count: int = Field(default=0, ge=0)
    missing_information_need_count: int = Field(default=0, ge=0)
    total_information_need_count: int = Field(default=0, ge=0)
    unresolved_information: list[str] = Field(default_factory=list)
    rationale: str
    grader_name: str
    fallback_used: bool = False
    grades: list[EvidenceGradeResponse] = Field(default_factory=list)
    information_need_grades: list[InformationNeedGradeResponse] = Field(default_factory=list)


class ClaimRetrievalTaskResponse(BaseModel):
    information_need_id: str
    description: str
    retrieval_query: str
    prior_status: str
    prior_coverage_score: float = Field(ge=0.0, le=1.0)
    prior_supporting_evidence_ranks: list[int] = Field(default_factory=list)
    grading_feedback: str
    rationale: str


class ClaimRetrievalPlanResponse(BaseModel):
    planner_name: str
    rationale: str
    target_information_need_ids: list[str] = Field(default_factory=list)
    target_information_need_count: int = Field(default=0, ge=0)
    deferred_information_need_ids: list[str] = Field(default_factory=list)
    deferred_information_need_count: int = Field(default=0, ge=0)
    tasks: list[ClaimRetrievalTaskResponse] = Field(default_factory=list)


class ClaimLookupResponse(BaseModel):
    information_need_id: str
    query: str
    pipeline_name: str
    strategy: str
    top_k: int = Field(ge=1)
    retrieved_count: int = Field(ge=0)
    unique_evidence_added: int = Field(ge=0)


class RetrievalAttemptResponse(BaseModel):
    attempt_number: int = Field(ge=1)
    retry_number: int = Field(ge=0)
    query: str
    top_k: int = Field(ge=1)
    pipeline_name: str
    strategy: str
    evidence_count: int = Field(ge=0)
    new_evidence_count: int = Field(default=0, ge=0)
    accumulated_evidence_count: int = Field(default=0, ge=0)
    actions: list[str] = Field(default_factory=list)
    decision_rationale: str | None = None
    resolved_information_need_ids: list[str] = Field(default_factory=list)
    remaining_information_need_ids: list[str] = Field(default_factory=list)
    claim_retrieval_plan: ClaimRetrievalPlanResponse | None = None
    claim_lookups: list[ClaimLookupResponse] = Field(default_factory=list)
    retrieval_plan: RetrievalPlanResponse
    evidence_grading: EvidenceGradingResponse


class RetrievalRetryResponse(BaseModel):
    policy_name: str
    max_retries: int = Field(ge=0)
    retries_used: int = Field(ge=0)
    attempt_count: int = Field(ge=1)
    claim_plan_count: int = Field(default=0, ge=0)
    claim_lookup_count: int = Field(default=0, ge=0)
    stop_reason: str
    stop_rationale: str
    final_sufficient: bool
    final_pipeline_name: str
    final_strategy: str
    final_top_k: int = Field(ge=1)
    final_query: str
    query_changed: bool
    attempts: list[RetrievalAttemptResponse] = Field(default_factory=list)


class InformationNeedRetrievalPlanResponse(BaseModel):
    information_need_id: str
    strategy: str
    selected_pipeline_name: str
    query: str
    top_k: int = Field(ge=1)
    rationale: str
    planner_name: str
    based_on_query_type: str
    attempt_number: int = Field(ge=1)
    metadata_filter_hints: list[str] = Field(default_factory=list)
    requires_reranking: bool = False
    adjustments: list[str] = Field(default_factory=list)
    document_constraint: DocumentNameConstraintResponse = Field(default_factory=DocumentNameConstraintResponse)
    version_constraint: VersionConstraintResponse = Field(default_factory=VersionConstraintResponse)
    date_constraints: list[DateConstraintResponse] = Field(default_factory=list)


class InformationNeedAttemptResponse(BaseModel):
    attempt_number: int = Field(ge=1)
    query: str
    top_k: int = Field(ge=1)
    pipeline_name: str
    strategy: str
    adjustments: list[str] = Field(default_factory=list)
    retrieved_count: int = Field(ge=0)
    unique_evidence_added: int = Field(ge=0)
    evidence_keys: list[str] = Field(default_factory=list)
    plan: InformationNeedRetrievalPlanResponse
    constraint_validation: ConstraintValidationResponse
    evidence_grading: EvidenceGradingResponse


class InformationNeedExecutionResponse(BaseModel):
    information_need: InformationNeedResponse
    information_need_id: str
    status: str
    attempts_used: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    reclassifications_used: int = Field(ge=0)
    parent_information_need_id: str | None = None
    depth: int = Field(ge=0)
    classification: QueryClassificationResponse | None = None
    classification_history: list[QueryClassificationResponse] = Field(default_factory=list)
    current_plan: InformationNeedRetrievalPlanResponse | None = None
    plan_history: list[InformationNeedRetrievalPlanResponse] = Field(default_factory=list)
    attempts: list[InformationNeedAttemptResponse] = Field(default_factory=list)
    constraint_validation_history: list[ConstraintValidationResponse] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)
    final_grade: InformationNeedGradeResponse | None = None
    stop_reason: str | None = None
    stop_rationale: str | None = None


class InformationNeedResolutionResponse(BaseModel):
    graph_name: str
    information_need_count: int = Field(ge=1)
    supported_information_need_ids: list[str] = Field(default_factory=list)
    supported_information_need_count: int = Field(ge=0)
    unresolved_information_need_ids: list[str] = Field(default_factory=list)
    unresolved_information_need_count: int = Field(ge=0)
    complete: bool
    total_retrieval_attempts: int = Field(ge=0)
    max_total_retrieval_attempts: int = Field(ge=1)
    max_attempts_per_information_need: int = Field(ge=1)
    executions: list[InformationNeedExecutionResponse] = Field(default_factory=list)


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
    information_need_decomposition: InformationNeedDecompositionResponse | None = None
    retrieval_plan: RetrievalPlanResponse | None = None
    evidence_grading: EvidenceGradingResponse | None = None
    retrieval_retry: RetrievalRetryResponse | None = None
    information_need_resolution: InformationNeedResolutionResponse | None = None
    constraint_validation: ConstraintValidationResponse | None = None
    evidence_context: EvidenceContextResponse | None = None
    evidence: list[EvidenceResponse] = Field(default_factory=list)
    citations: list[CitationResponse] = Field(default_factory=list)
    trace: list[TraceStepResponse] = Field(default_factory=list)
