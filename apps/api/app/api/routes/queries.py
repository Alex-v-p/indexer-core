from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.composition import build_query_graph
from app.core.config import Settings, get_settings
from app.dependencies.database import get_unit_of_work
from app.schemas.queries import CitationResponse, EvidenceResponse, QueryRequest, QueryResponse, TraceStepResponse
from packages.indexer_application.dto import QueryRunRecord
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.services import get_query_run, run_query
from packages.rag_core.pipelines import UnknownPipelineError

router = APIRouter(prefix="/queries", tags=["queries"])


@router.post("", response_model=QueryResponse, status_code=status.HTTP_201_CREATED)
async def create_query_run(
    payload: QueryRequest,
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    try:
        pipeline = build_query_graph(settings, pipeline_name=payload.pipeline_name)
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
