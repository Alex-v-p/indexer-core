from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from app.dependencies.database import get_unit_of_work
from app.dependencies.query_runtime import get_query_pipeline_registry
from app.schemas.queries import (
    CitationResponse,
    ConstraintValidationResponse,
    EvidenceContextResponse,
    EvidenceGradingResponse,
    EvidenceResponse,
    InformationNeedDecompositionResponse,
    InformationNeedResolutionResponse,
    QueryClassificationResponse,
    QueryRequest,
    QueryResponse,
    RetrievalPlanResponse,
    RetrievalRetryResponse,
    TraceStepResponse,
)
from packages.indexer_application.dto import QueryRunRecord
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.services import get_query_run, run_query
from packages.rag_core.pipelines import PipelineRegistry, UnknownPipelineError

router = APIRouter(prefix="/queries", tags=["queries"])


@router.post("", response_model=QueryResponse, status_code=status.HTTP_201_CREATED)
async def create_query_run(
    payload: QueryRequest,
    uow: UnitOfWork = Depends(get_unit_of_work),
    pipeline_registry: PipelineRegistry = Depends(get_query_pipeline_registry),
) -> QueryResponse:
    try:
        pipeline = pipeline_registry.build(payload.pipeline_name)
        query_run = await run_query(
            uow=uow,
            pipeline=pipeline,
            question=payload.question,
            top_k=payload.top_k,
            requested_pipeline_name=payload.pipeline_name,
        )
    except UnknownPipelineError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return to_query_response(query_run)


@router.get("/{query_run_id}", response_model=QueryResponse)
async def read_query_run(
    query_run_id: uuid.UUID,
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> QueryResponse:
    query_run = await get_query_run(uow=uow, query_run_id=query_run_id)
    if query_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query run not found.")
    return to_query_response(query_run)


def to_query_response(query_run: QueryRunRecord) -> QueryResponse:
    return QueryResponse(
        id=query_run.id,
        question=query_run.question,
        answer=query_run.answer,
        status=query_run.status.value,
        pipeline_name=query_run.pipeline_name,
        pipeline_version=query_run.pipeline_version,
        top_k=query_run.top_k,
        started_at=query_run.started_at,
        completed_at=query_run.completed_at,
        error_message=query_run.error_message,
        classification=_to_classification_response(query_run.metadata),
        information_need_decomposition=_to_information_need_decomposition_response(query_run.metadata),
        retrieval_plan=_to_retrieval_plan_response(query_run.metadata),
        evidence_grading=_to_evidence_grading_response(query_run.metadata),
        retrieval_retry=_to_retrieval_retry_response(query_run.metadata),
        information_need_resolution=_to_information_need_resolution_response(query_run.metadata),
        constraint_validation=_to_constraint_validation_response(query_run.metadata),
        evidence_context=_to_evidence_context_response(query_run.metadata),
        evidence=[
            EvidenceResponse(
                id=item.id,
                rank=item.rank,
                score=item.score,
                text=item.text,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
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


def _to_classification_response(metadata: dict[str, object]) -> QueryClassificationResponse | None:
    value = metadata.get("query_classification")
    if not isinstance(value, dict):
        return None
    try:
        return QueryClassificationResponse.model_validate(value)
    except ValidationError:
        return None


def _to_information_need_decomposition_response(
    metadata: dict[str, object],
) -> InformationNeedDecompositionResponse | None:
    value = metadata.get("information_need_decomposition")
    if not isinstance(value, dict):
        return None
    try:
        return InformationNeedDecompositionResponse.model_validate(value)
    except ValidationError:
        return None


def _to_retrieval_plan_response(metadata: dict[str, object]) -> RetrievalPlanResponse | None:
    value = metadata.get("retrieval_plan")
    if not isinstance(value, dict):
        return None
    try:
        return RetrievalPlanResponse.model_validate(value)
    except ValidationError:
        return None


def _to_evidence_grading_response(metadata: dict[str, object]) -> EvidenceGradingResponse | None:
    value = metadata.get("evidence_grading")
    if not isinstance(value, dict):
        return None
    try:
        return EvidenceGradingResponse.model_validate(value)
    except ValidationError:
        return None


def _to_retrieval_retry_response(metadata: dict[str, object]) -> RetrievalRetryResponse | None:
    value = metadata.get("retrieval_retry")
    if not isinstance(value, dict):
        return None
    try:
        return RetrievalRetryResponse.model_validate(value)
    except ValidationError:
        return None


def _to_information_need_resolution_response(
    metadata: dict[str, object],
) -> InformationNeedResolutionResponse | None:
    value = metadata.get("information_need_resolution")
    if not isinstance(value, dict):
        return None
    try:
        return InformationNeedResolutionResponse.model_validate(value)
    except ValidationError:
        return None


def _to_constraint_validation_response(
    metadata: dict[str, object],
) -> ConstraintValidationResponse | None:
    value = metadata.get("constraint_validation")
    if not isinstance(value, dict):
        return None
    try:
        return ConstraintValidationResponse.model_validate(value)
    except ValidationError:
        return None


def _to_evidence_context_response(
    metadata: dict[str, object],
) -> EvidenceContextResponse | None:
    value = metadata.get("evidence_context")
    if not isinstance(value, dict):
        return None
    try:
        return EvidenceContextResponse.model_validate(value)
    except ValidationError:
        return None
