from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.schemas.queries import (
    AnswerPresentationResponse,
    CitationResponse,
    ConstraintValidationResponse,
    DocumentPreferenceResponse,
    EvidenceContextResponse,
    EvidenceGradingResponse,
    EvidenceResponse,
    InformationNeedDecompositionResponse,
    InformationNeedResolutionResponse,
    QueryClassificationResponse,
    QueryResponse,
    ResolvedSubjectScopeResponse,
    RetrievalPlanResponse,
    RetrievalRetryResponse,
    TraceStepResponse,
)
from packages.indexer_application.dto import QueryExecutionResult, QueryRunRecord

def to_query_response(
    query_result: QueryExecutionResult | QueryRunRecord,
) -> QueryResponse:
    result = (
        query_result
        if isinstance(query_result, QueryExecutionResult)
        else QueryExecutionResult.from_record(query_result)
    )
    query_run = result.query_run
    metadata = result.metadata
    return QueryResponse(
        id=query_run.id,
        question=query_run.question,
        answer=query_run.answer,
        answer_presentation=_to_answer_presentation_response(metadata.answer_presentation),
        status=query_run.status.value,
        pipeline_name=query_run.pipeline_name,
        pipeline_version=query_run.pipeline_version,
        top_k=query_run.top_k,
        started_at=query_run.started_at,
        completed_at=query_run.completed_at,
        error_message=query_run.error_message,
        product_outcome=metadata.product_outcome,
        subject_scope=_validate_payload(
            ResolvedSubjectScopeResponse,
            metadata.resolved_subject_scope,
        ),
        classification=_to_classification_payload(metadata.classification),
        information_need_decomposition=_to_information_need_decomposition_payload(
            metadata.information_need_decomposition,
        ),
        retrieval_plan=_to_retrieval_plan_payload(metadata.retrieval_plan),
        evidence_grading=_to_evidence_grading_payload(metadata.evidence_grading),
        retrieval_retry=_to_retrieval_retry_payload(metadata.retrieval_retry),
        primary_document_preference=_to_primary_document_preference_payload(
            metadata.primary_document_preference,
        ),
        information_need_resolution=_to_information_need_resolution_payload(
            metadata.information_need_resolution,
        ),
        constraint_validation=_to_constraint_validation_payload(metadata.constraint_validation),
        evidence_context=_to_evidence_context_payload(metadata.evidence_context),
        evidence=[
            EvidenceResponse(
                id=item.id,
                rank=item.rank,
                score=item.score,
                text=item.text,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                subject_lane_id=_lane_value(item.metadata, "lane_id"),
                subject_id=_lane_value(item.metadata, "subject_id"),
                subject_name=_lane_value(item.metadata, "subject_name"),
                metadata=item.metadata,
            )
            for item in query_run.evidence_items
        ],
        citations=[
            CitationResponse(
                id=item.id,
                citation_index=item.citation_index,
                label=item.label,
                evidence_id=item.evidence_id,
                page_number=item.page_number,
                quote=item.quote,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                subject_lane_id=_lane_value(item.metadata, "lane_id"),
                subject_id=_lane_value(item.metadata, "subject_id"),
                subject_name=_lane_value(item.metadata, "subject_name"),
                metadata=item.metadata,
            )
            for item in query_run.citations
        ],
        trace=[
            TraceStepResponse(
                id=item.id,
                step_order=item.step_order,
                name=item.name,
                step_type=item.step_type,
                status=item.status.value,
                duration_ms=item.duration_ms,
                input_summary=item.input_summary,
                output_summary=item.output_summary,
                error_message=item.error_message,
                metadata=item.metadata,
            )
            for item in query_run.trace_steps
        ],
    )


def _to_answer_presentation_response(
    metadata_or_payload: dict[str, object] | None,
) -> AnswerPresentationResponse | None:
    payload = _legacy_metadata_payload(metadata_or_payload, "answer_presentation")
    return _validate_payload(AnswerPresentationResponse, payload)


def _to_classification_response(metadata: dict[str, object]) -> QueryClassificationResponse | None:
    return _to_classification_payload(_legacy_metadata_payload(metadata, "query_classification"))


def _to_classification_payload(payload: dict[str, object] | None) -> QueryClassificationResponse | None:
    return _validate_payload(QueryClassificationResponse, payload)


def _to_information_need_decomposition_response(
    metadata: dict[str, object],
) -> InformationNeedDecompositionResponse | None:
    return _to_information_need_decomposition_payload(
        _legacy_metadata_payload(metadata, "information_need_decomposition"),
    )


def _to_information_need_decomposition_payload(
    payload: dict[str, object] | None,
) -> InformationNeedDecompositionResponse | None:
    return _validate_payload(InformationNeedDecompositionResponse, payload)


def _to_retrieval_plan_response(metadata: dict[str, object]) -> RetrievalPlanResponse | None:
    return _to_retrieval_plan_payload(_legacy_metadata_payload(metadata, "retrieval_plan"))


def _to_retrieval_plan_payload(payload: dict[str, object] | None) -> RetrievalPlanResponse | None:
    return _validate_payload(RetrievalPlanResponse, payload)


def _to_evidence_grading_response(metadata: dict[str, object]) -> EvidenceGradingResponse | None:
    return _to_evidence_grading_payload(_legacy_metadata_payload(metadata, "evidence_grading"))


def _to_evidence_grading_payload(payload: dict[str, object] | None) -> EvidenceGradingResponse | None:
    return _validate_payload(EvidenceGradingResponse, payload)


def _to_retrieval_retry_response(metadata: dict[str, object]) -> RetrievalRetryResponse | None:
    return _to_retrieval_retry_payload(_legacy_metadata_payload(metadata, "retrieval_retry"))


def _to_retrieval_retry_payload(payload: dict[str, object] | None) -> RetrievalRetryResponse | None:
    return _validate_payload(RetrievalRetryResponse, payload)


def _to_primary_document_preference_response(
    metadata: dict[str, object],
) -> DocumentPreferenceResponse | None:
    return _to_primary_document_preference_payload(
        _legacy_metadata_payload(metadata, "primary_document_preference"),
    )


def _to_primary_document_preference_payload(
    payload: dict[str, object] | None,
) -> DocumentPreferenceResponse | None:
    return _validate_payload(DocumentPreferenceResponse, payload)


def _to_information_need_resolution_response(
    metadata: dict[str, object],
) -> InformationNeedResolutionResponse | None:
    return _to_information_need_resolution_payload(
        _legacy_metadata_payload(metadata, "information_need_resolution"),
    )


def _to_information_need_resolution_payload(
    payload: dict[str, object] | None,
) -> InformationNeedResolutionResponse | None:
    return _validate_payload(InformationNeedResolutionResponse, payload)


def _to_constraint_validation_response(
    metadata: dict[str, object],
) -> ConstraintValidationResponse | None:
    return _to_constraint_validation_payload(
        _legacy_metadata_payload(metadata, "constraint_validation"),
    )


def _to_constraint_validation_payload(
    payload: dict[str, object] | None,
) -> ConstraintValidationResponse | None:
    return _validate_payload(ConstraintValidationResponse, payload)


def _to_evidence_context_response(
    metadata: dict[str, object],
) -> EvidenceContextResponse | None:
    return _to_evidence_context_payload(_legacy_metadata_payload(metadata, "evidence_context"))


def _to_evidence_context_payload(payload: dict[str, object] | None) -> EvidenceContextResponse | None:
    return _validate_payload(EvidenceContextResponse, payload)


def _legacy_metadata_payload(
    metadata_or_payload: dict[str, object] | None,
    key: str,
) -> dict[str, object] | None:
    if metadata_or_payload is None:
        return None
    nested = metadata_or_payload.get(key)
    if isinstance(nested, dict):
        return nested
    return metadata_or_payload


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


def _validate_payload(
    model_type: type[ResponseModelT],
    payload: dict[str, object] | None,
) -> ResponseModelT | None:
    if payload is None:
        return None
    try:
        return model_type.model_validate(payload)
    except ValidationError:
        return None


def _lane_value(metadata: dict[str, object], key: str):
    lane = metadata.get("subject_lane")
    return lane.get(key) if isinstance(lane, dict) else None
