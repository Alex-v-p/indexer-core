from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.database.models import Citation, Evidence, QueryRun, QueryRunStatus, TraceStep, TraceStepStatus
from app.core.config import Settings
from app.services.query_graph import build_query_graph
from packages.rag_core.query import CitationItem, QueryState, TraceEvent


async def run_query(*, session: AsyncSession, settings: Settings, question: str, top_k: int) -> QueryRun:
    """Persist and execute a query through the graph runner."""

    graph = build_query_graph(settings)
    query_run = QueryRun(
        question=question,
        status=QueryRunStatus.RUNNING,
        pipeline_name=graph.name,
        pipeline_version=graph.version,
        top_k=top_k,
        metadata_={"runner": "graph"},
    )
    session.add(query_run)
    await session.flush()

    state = QueryState(
        question=question,
        top_k=top_k,
        query_run_id=query_run.id,
        pipeline_name=graph.name,
        pipeline_version=graph.version,
    )

    try:
        state = await graph.run(state)
    except Exception as exc:
        query_run.status = QueryRunStatus.FAILED
        query_run.completed_at = datetime.now(UTC)
        query_run.error_message = str(exc)
        _add_trace_steps(query_run.id, state.trace, session)
        await session.commit()
        return await get_query_run(session=session, query_run_id=query_run.id) or query_run

    query_run.answer = state.answer
    query_run.status = QueryRunStatus.SUCCEEDED
    query_run.completed_at = datetime.now(UTC)
    query_run.pipeline_name = state.pipeline_name
    query_run.pipeline_version = state.pipeline_version
    query_run.metadata_ = {**(query_run.metadata_ or {}), **state.metadata}

    evidence_by_rank = await _add_evidence(query_run.id, state, session)
    await session.flush()
    _add_citations(query_run.id, state.citations, evidence_by_rank, session)
    _add_trace_steps(query_run.id, state.trace, session)

    await session.commit()
    return await get_query_run(session=session, query_run_id=query_run.id) or query_run


async def get_query_run(*, session: AsyncSession, query_run_id: uuid.UUID) -> QueryRun | None:
    statement = (
        select(QueryRun)
        .where(QueryRun.id == query_run_id)
        .options(
            selectinload(QueryRun.evidence_items),
            selectinload(QueryRun.citations),
            selectinload(QueryRun.trace_steps),
        )
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def _add_evidence(query_run_id: uuid.UUID, state: QueryState, session: AsyncSession) -> dict[int, Evidence]:
    evidence_by_rank: dict[int, Evidence] = {}
    for item in state.retrieved_evidence:
        evidence = Evidence(
            query_run_id=query_run_id,
            qdrant_chunk_index_id=item.qdrant_chunk_index_id,
            document_id=item.document_id,
            document_version_id=item.document_version_id,
            rank=item.rank,
            score=item.score,
            text=item.text,
            metadata_=item.metadata,
        )
        session.add(evidence)
        evidence_by_rank[item.rank] = evidence
    return evidence_by_rank


def _add_citations(
    query_run_id: uuid.UUID,
    citations: list[CitationItem],
    evidence_by_rank: dict[int, Evidence],
    session: AsyncSession,
) -> None:
    for item in citations:
        evidence = evidence_by_rank.get(item.evidence_rank or -1)
        session.add(
            Citation(
                query_run_id=query_run_id,
                evidence_id=evidence.id if evidence else None,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                citation_index=item.citation_index,
                label=item.label,
                page_number=item.page_number,
                quote=item.quote,
                metadata_=item.metadata,
            ),
        )


def _add_trace_steps(query_run_id: uuid.UUID, trace: list[TraceEvent], session: AsyncSession) -> None:
    for item in trace:
        session.add(
            TraceStep(
                query_run_id=query_run_id,
                step_order=item.step_order,
                name=item.name,
                step_type=item.step_type,
                status=TraceStepStatus(item.status),
                duration_ms=item.duration_ms,
                input_summary=item.input_summary,
                output_summary=item.output_summary,
                error_message=item.error_message,
                metadata_=item.metadata,
            ),
        )
