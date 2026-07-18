from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.indexer_application.dto import (
    ChunkIndexCreate,
    ChunkIndexRecord,
    CitationRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentVersionIdentity,
    DocumentVersionRecord,
    DocumentVersionStatus,
    EvidenceRecord,
    QueryRunRecord,
    QueryRunStatus,
    TraceStepRecord,
    TraceStepStatus,
)
from packages.indexer_application.ports.object_storage import StoredDocumentFile
from packages.indexer_infrastructure.postgres.models import (
    Citation,
    Document,
    DocumentVersion,
    Evidence,
    QdrantChunkIndex,
    QueryRun,
    TraceStep,
)
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.runtime import TraceEvent
from packages.rag_core.documents.version_families import (
    document_family_key,
    normalized_document_identity,
)


class SqlAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_processing_document(self, *, stored_file: StoredDocumentFile, title: str) -> uuid.UUID:
        document = Document(
            title=title,
            original_filename=stored_file.original_filename,
            content_type=stored_file.content_type,
            storage_uri=stored_file.storage_uri,
            size_bytes=stored_file.size_bytes,
            checksum_sha256=stored_file.checksum_sha256,
            status=DocumentStatus.PROCESSING,
            metadata_=_storage_metadata(stored_file),
        )
        self._session.add(document)
        await self._session.flush()
        return document.id

    async def create_processing_version(
        self,
        *,
        document_id: uuid.UUID,
        stored_file: StoredDocumentFile,
        published_at: datetime | None = None,
    ) -> DocumentVersionIdentity:
        lock_statement = select(Document.id).where(Document.id == document_id).with_for_update()
        lock_result = await self._session.execute(lock_statement)
        if lock_result.scalar_one_or_none() is None:
            raise LookupError(f"Document {document_id} was not found.")

        statement = select(func.coalesce(func.max(DocumentVersion.version_number), 0)).where(
            DocumentVersion.document_id == document_id,
        )
        result = await self._session.execute(statement)
        uploaded_at = datetime.now(UTC)
        version = DocumentVersion(
            document_id=document_id,
            version_number=int(result.scalar_one()) + 1,
            storage_uri=stored_file.storage_uri,
            content_type=stored_file.content_type,
            checksum_sha256=stored_file.checksum_sha256,
            published_at=published_at,
            status=DocumentVersionStatus.PROCESSING,
            created_at=uploaded_at,
            metadata_=_storage_metadata(stored_file),
        )
        self._session.add(version)
        await self._session.flush()
        return DocumentVersionIdentity(
            id=version.id,
            version_number=version.version_number,
            uploaded_at=uploaded_at,
            published_at=published_at,
        )

    async def find_version_candidate(
        self,
        *,
        title: str,
        original_filename: str,
    ) -> DocumentRecord | None:
        normalized_title = normalized_document_identity(title)
        normalized_filename = normalized_document_identity(original_filename)
        title_family = document_family_key(title)
        filename_family = document_family_key(original_filename)

        statement = (
            select(Document)
            .order_by(Document.updated_at.desc())
            .limit(250)
            .options(selectinload(Document.versions), selectinload(Document.qdrant_chunk_indexes))
        )
        result = await self._session.execute(statement)
        candidates = list(result.scalars().unique().all())
        if not candidates:
            return None

        for item in candidates:
            if (
                normalized_document_identity(item.title) == normalized_title
                or normalized_document_identity(item.original_filename or "") == normalized_filename
            ):
                return _to_document_record(item)

        family_candidates = []
        for item in candidates:
            candidate_keys = {
                document_family_key(item.title),
                document_family_key(item.original_filename or ""),
            }
            requested_keys = {key for key in (title_family, filename_family) if key and len(key) >= 4}
            if requested_keys.intersection(candidate_keys):
                family_candidates.append(item)

        if len(family_candidates) == 1:
            return _to_document_record(family_candidates[0])
        return None

    async def set_version_parser_metadata(
        self,
        *,
        version_id: uuid.UUID,
        parser_name: str,
        parser_version: str,
        metadata: dict[str, Any],
    ) -> None:
        version = await self._require_version(version_id)
        version.parser_name = parser_name
        version.parser_version = parser_version
        version.metadata_ = {**(version.metadata_ or {}), **metadata}

    async def add_chunk_indexes(self, chunks: list[ChunkIndexCreate]) -> None:
        self._session.add_all(
            [
                QdrantChunkIndex(
                    id=item.id,
                    document_id=item.document_id,
                    document_version_id=item.document_version_id,
                    ordinal=item.ordinal,
                    content_hash=item.content_hash,
                    token_count=item.token_count,
                    source_page_start=item.source_page_start,
                    source_page_end=item.source_page_end,
                    section_title=item.section_title,
                    qdrant_collection=item.qdrant_collection,
                    qdrant_point_id=item.qdrant_point_id,
                    metadata_=item.metadata,
                )
                for item in chunks
            ],
        )

    async def mark_ready(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        document_metadata: dict[str, Any],
        stored_file: StoredDocumentFile,
        version_number: int,
    ) -> None:
        document = await self._require_document(document_id)
        version = await self._require_version(version_id)
        document.status = DocumentStatus.READY
        document.original_filename = stored_file.original_filename
        document.content_type = stored_file.content_type
        document.storage_uri = stored_file.storage_uri
        document.size_bytes = stored_file.size_bytes
        document.checksum_sha256 = stored_file.checksum_sha256
        document.metadata_ = {
            **(document.metadata_ or {}),
            **document_metadata,
            "latest_version_number": version_number,
        }
        version.status = DocumentVersionStatus.READY

    async def mark_failed(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        error_message: str,
    ) -> None:
        document = await self._require_document(document_id)
        version = await self._require_version(version_id)
        version.status = DocumentVersionStatus.FAILED
        version.metadata_ = {**(version.metadata_ or {}), "error_message": error_message}
        ready_statement = select(func.count(DocumentVersion.id)).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.status == DocumentVersionStatus.READY,
        )
        ready_result = await self._session.execute(ready_statement)
        has_ready_version = int(ready_result.scalar_one()) > 0
        document.status = DocumentStatus.READY if has_ready_version else DocumentStatus.FAILED
        document.metadata_ = {**(document.metadata_ or {}), "latest_ingestion_error": error_message}

    async def get(self, document_id: uuid.UUID) -> DocumentRecord | None:
        statement = (
            select(Document)
            .where(Document.id == document_id)
            .options(selectinload(Document.versions), selectinload(Document.qdrant_chunk_indexes))
        )
        result = await self._session.execute(statement)
        model = result.scalar_one_or_none()
        return _to_document_record(model) if model else None

    async def list(self, *, limit: int, offset: int) -> list[DocumentRecord]:
        statement = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
            .options(selectinload(Document.versions), selectinload(Document.qdrant_chunk_indexes))
        )
        result = await self._session.execute(statement)
        return [_to_document_record(model) for model in result.scalars().unique().all()]

    async def _require_document(self, document_id: uuid.UUID) -> Document:
        document = await self._session.get(Document, document_id)
        if document is None:
            raise LookupError(f"Document {document_id} was not found.")
        return document

    async def _require_version(self, version_id: uuid.UUID) -> DocumentVersion:
        version = await self._session.get(DocumentVersion, version_id)
        if version is None:
            raise LookupError(f"Document version {version_id} was not found.")
        return version


class SqlAlchemyQueryRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_running(
        self,
        *,
        question: str,
        pipeline_name: str,
        pipeline_version: str,
        top_k: int,
        requested_pipeline_name: str | None,
    ) -> uuid.UUID:
        query_run = QueryRun(
            question=question,
            status=QueryRunStatus.RUNNING,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            top_k=top_k,
            metadata_={"runner": "graph", "requested_pipeline_name": requested_pipeline_name},
        )
        self._session.add(query_run)
        await self._session.flush()
        return query_run.id

    async def mark_failed(
        self,
        *,
        query_run_id: uuid.UUID,
        error_message: str,
        trace: list[TraceEvent],
    ) -> None:
        query_run = await self._require_query_run(query_run_id)
        query_run.status = QueryRunStatus.FAILED
        query_run.completed_at = datetime.now(UTC)
        query_run.error_message = error_message
        self._add_trace_steps(query_run_id, trace)

    async def mark_succeeded(self, *, query_run_id: uuid.UUID, state: QueryState) -> None:
        query_run = await self._require_query_run(query_run_id)
        query_run.answer = state.answer
        query_run.status = QueryRunStatus.SUCCEEDED
        query_run.completed_at = datetime.now(UTC)
        query_run.pipeline_name = state.pipeline_name
        query_run.pipeline_version = state.pipeline_version
        query_run.metadata_ = {**(query_run.metadata_ or {}), **state.metadata}

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
            self._session.add(evidence)
            evidence_by_rank[item.rank] = evidence
        await self._session.flush()

        for item in state.citations:
            evidence = evidence_by_rank.get(item.evidence_rank or -1)
            self._session.add(
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
        self._add_trace_steps(query_run_id, state.trace)

    async def get(self, query_run_id: uuid.UUID) -> QueryRunRecord | None:
        statement = (
            select(QueryRun)
            .where(QueryRun.id == query_run_id)
            .options(
                selectinload(QueryRun.evidence_items),
                selectinload(QueryRun.citations),
                selectinload(QueryRun.trace_steps),
            )
        )
        result = await self._session.execute(statement)
        model = result.scalar_one_or_none()
        return _to_query_run_record(model) if model else None

    async def _require_query_run(self, query_run_id: uuid.UUID) -> QueryRun:
        query_run = await self._session.get(QueryRun, query_run_id)
        if query_run is None:
            raise LookupError(f"Query run {query_run_id} was not found.")
        return query_run

    def _add_trace_steps(self, query_run_id: uuid.UUID, trace: list[TraceEvent]) -> None:
        for item in trace:
            self._session.add(
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


def _storage_metadata(stored_file: StoredDocumentFile) -> dict[str, str | None]:
    return {
        "storage_backend": stored_file.storage_backend,
        "bucket_name": stored_file.bucket_name,
        "object_key": stored_file.object_key,
        "storage_uri": stored_file.storage_uri,
    }


def _to_document_record(model: Document) -> DocumentRecord:
    return DocumentRecord(
        id=model.id,
        title=model.title,
        original_filename=model.original_filename,
        content_type=model.content_type,
        storage_uri=model.storage_uri,
        size_bytes=model.size_bytes,
        checksum_sha256=model.checksum_sha256,
        status=model.status,
        metadata=dict(model.metadata_ or {}),
        created_at=model.created_at,
        updated_at=model.updated_at,
        versions=tuple(
            DocumentVersionRecord(
                id=item.id,
                version_number=item.version_number,
                storage_uri=item.storage_uri,
                content_type=item.content_type,
                checksum_sha256=item.checksum_sha256,
                parser_name=item.parser_name,
                parser_version=item.parser_version,
                status=item.status,
                metadata=dict(item.metadata_ or {}),
                published_at=item.published_at,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in model.versions
        ),
        chunk_indexes=tuple(
            ChunkIndexRecord(
                id=item.id,
                ordinal=item.ordinal,
                content_hash=item.content_hash,
                token_count=item.token_count,
                source_page_start=item.source_page_start,
                source_page_end=item.source_page_end,
                section_title=item.section_title,
                qdrant_collection=item.qdrant_collection,
                qdrant_point_id=item.qdrant_point_id,
                metadata=dict(item.metadata_ or {}),
                created_at=item.created_at,
            )
            for item in model.qdrant_chunk_indexes
        ),
    )


def _to_query_run_record(model: QueryRun) -> QueryRunRecord:
    return QueryRunRecord(
        id=model.id,
        question=model.question,
        answer=model.answer,
        status=model.status,
        pipeline_name=model.pipeline_name,
        pipeline_version=model.pipeline_version,
        top_k=model.top_k,
        started_at=model.started_at,
        completed_at=model.completed_at,
        error_message=model.error_message,
        metadata=dict(model.metadata_ or {}),
        evidence_items=tuple(
            EvidenceRecord(
                id=item.id,
                rank=item.rank,
                score=item.score,
                text=item.text,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                metadata=dict(item.metadata_ or {}),
            )
            for item in model.evidence_items
        ),
        citations=tuple(
            CitationRecord(
                id=item.id,
                citation_index=item.citation_index,
                label=item.label,
                evidence_id=item.evidence_id,
                page_number=item.page_number,
                quote=item.quote,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                metadata=dict(item.metadata_ or {}),
            )
            for item in model.citations
        ),
        trace_steps=tuple(
            TraceStepRecord(
                id=item.id,
                step_order=item.step_order,
                name=item.name,
                step_type=item.step_type,
                status=item.status,
                duration_ms=item.duration_ms,
                input_summary=item.input_summary,
                output_summary=item.output_summary,
                error_message=item.error_message,
                metadata=dict(item.metadata_ or {}),
            )
            for item in model.trace_steps
        ),
    )
