from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.indexer_application.dto import (
    ChunkIndexCreate,
    DocumentRecord,
    DocumentStatus,
    DocumentVersionIdentity,
    DocumentVersionStatus,
)
from packages.indexer_application.ports.object_storage import StoredDocumentReference
from packages.indexer_infrastructure.postgres.models import (
    Document,
    DocumentVersion,
    QdrantChunkIndex,
)
from packages.indexer_infrastructure.postgres.repositories.mappers import (
    storage_metadata,
    to_document_record,
)
from packages.rag_core.documents.version_families import (
    document_family_key,
    normalized_document_identity,
)


class SqlAlchemyDocumentRepository:
    """SQLAlchemy adapter for the document and document-version aggregate."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_processing_document(
        self,
        *,
        stored_file: StoredDocumentReference,
        title: str,
    ) -> uuid.UUID:
        document = Document(
            title=title,
            original_filename=stored_file.original_filename,
            content_type=stored_file.content_type,
            storage_uri=stored_file.storage_uri,
            size_bytes=stored_file.size_bytes,
            checksum_sha256=stored_file.checksum_sha256,
            status=DocumentStatus.PROCESSING,
            metadata_=storage_metadata(stored_file),
        )
        self._session.add(document)
        await self._session.flush()
        return document.id

    async def create_processing_version(
        self,
        *,
        document_id: uuid.UUID,
        stored_file: StoredDocumentReference,
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
            metadata_=storage_metadata(stored_file),
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
                return to_document_record(item)

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
            return to_document_record(family_candidates[0])
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
        stored_file: StoredDocumentReference,
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
        return to_document_record(model) if model else None

    async def list(self, *, limit: int, offset: int) -> list[DocumentRecord]:
        statement = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
            .options(selectinload(Document.versions), selectinload(Document.qdrant_chunk_indexes))
        )
        result = await self._session.execute(statement)
        return [to_document_record(model) for model in result.scalars().unique().all()]

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
