from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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


class DocumentReferenceResponse(BaseModel):
    key: str
    display_name: str
    document_id: uuid.UUID | None = None
    document_version_ids: list[uuid.UUID] = Field(default_factory=list)
    normalized_names: list[str] = Field(default_factory=list)


class DocumentPreferenceResponse(BaseModel):
    document: DocumentReferenceResponse
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    margin: float = Field(ge=0.0, le=1.0)
    supporting_information_need_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ranks: list[int] = Field(default_factory=list)
    rationale: str
    detector_name: str
    semantics: str = "soft_preference_not_filter"


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


class StructuredOutputDiagnosticsResponse(BaseModel):
    schema_version: str = "1.0"
    outcome: str
    failure_code: str | None = None
    attempt_count: int = Field(ge=1, le=2)
    repair_attempted: bool = False


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
    structured_output: StructuredOutputDiagnosticsResponse | None = None


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
    structured_output: StructuredOutputDiagnosticsResponse | None = None


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
    preferred_document: DocumentPreferenceResponse | None = None


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
    structured_output: StructuredOutputDiagnosticsResponse | None = None
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
    preferred_document: DocumentPreferenceResponse | None = None


class InformationNeedAttemptEvidenceResponse(BaseModel):
    evidence_key: str
    retrieval_order: int = Field(ge=1)
    aggregate_rank: int | None = Field(default=None, ge=1)
    text: str
    score: float | None = None
    qdrant_chunk_index_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    relevant: bool | None = None
    grading_rationale: str | None = None
    supports_information_need_ids: list[str] = Field(default_factory=list)
    retained_after_need_grading: bool = False


class RetrievalExecutionMetadataResponse(BaseModel):
    requested_top_k: int | None = Field(default=None, ge=1)
    candidate_top_k: int | None = Field(default=None, ge=1)
    retriever_type: str | None = None
    fusion_method: str | None = None
    vector_contribution: float | None = None
    keyword_contribution: float | None = None
    query_variants: list[str] = Field(default_factory=list)
    multi_query_query_count: int | None = Field(default=None, ge=0)
    multi_query_candidate_top_k_per_query: int | None = Field(default=None, ge=1)
    multi_query_result_counts: dict[str, int] = Field(default_factory=dict)
    hierarchical_document_candidate_count: int | None = Field(default=None, ge=0)
    hierarchical_selected_document_version_ids: list[str] = Field(default_factory=list)
    hierarchical_section_candidate_count: int | None = Field(default=None, ge=0)
    hierarchical_selected_section_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class RerankingMetadataResponse(BaseModel):
    applied: bool = False
    provider: str | None = None
    candidate_count_before: int | None = Field(default=None, ge=0)
    candidate_count_after: int | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class AttemptDocumentBalancingResponse(BaseModel):
    selector_name: str | None = None
    candidate_count: int | None = Field(default=None, ge=0)
    requested_top_k: int | None = Field(default=None, ge=1)
    selected_count: int | None = Field(default=None, ge=0)
    selected_chunks_per_document: dict[str, int] = Field(default_factory=dict)
    primary_document_key: str | None = None
    primary_document_quota: int | None = Field(default=None, ge=0)
    primary_document_selected_count: int | None = Field(default=None, ge=0)
    quota_relaxed: bool = False
    preferred_document_active: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


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
    evidence: list[InformationNeedAttemptEvidenceResponse] = Field(default_factory=list)
    retrieval_metadata: RetrievalExecutionMetadataResponse = Field(
        default_factory=RetrievalExecutionMetadataResponse,
    )
    reranking_metadata: RerankingMetadataResponse = Field(
        default_factory=RerankingMetadataResponse,
    )
    document_balancing: AttemptDocumentBalancingResponse = Field(
        default_factory=AttemptDocumentBalancingResponse,
    )
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
    classification_source_history: list[Literal["top_level_reuse", "model"]] = Field(
        default_factory=list,
    )
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
