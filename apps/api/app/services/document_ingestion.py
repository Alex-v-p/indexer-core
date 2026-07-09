from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.database.models import Document, DocumentStatus, DocumentVersion, DocumentVersionStatus
from app.adapters.object_storage import DocumentStorageError, StoredDocumentFile, build_document_object_store
from app.core.config import Settings
from app.services.chunk_indexing import index_document_chunks
from packages.rag_core.documents import (
    ChunkingConfig,
    UnsupportedDocumentTypeError,
    chunk_document,
    is_supported_document,
    parse_document,
)


class IngestionError(RuntimeError):
    """Raised when an upload could not be ingested."""


async def ingest_uploaded_document(
    *,
    session: AsyncSession,
    settings: Settings,
    upload: UploadFile,
    title: str | None = None,
) -> Document:
    """Store, parse, chunk, embed, and index one uploaded document."""

    _validate_supported_upload(upload)

    object_store = build_document_object_store(settings)
    try:
        stored_file = await object_store.save_upload(upload)
    except DocumentStorageError as exc:
        raise IngestionError(str(exc)) from exc

    document = _create_document(stored_file=stored_file, title=title)
    session.add(document)
    await session.flush()

    version = DocumentVersion(
        document_id=document.id,
        version_number=await _next_version_number(session, document.id),
        storage_uri=stored_file.storage_uri,
        content_type=stored_file.content_type,
        checksum_sha256=stored_file.checksum_sha256,
        status=DocumentVersionStatus.PROCESSING,
        metadata_=_storage_metadata(stored_file),
    )
    session.add(version)
    await session.flush()

    try:
        parsed_document = parse_document(
            stored_file.path,
            filename=stored_file.original_filename,
            content_type=stored_file.content_type,
        )
        chunks = chunk_document(
            parsed_document,
            config=ChunkingConfig(max_chars=settings.chunk_max_chars, overlap_chars=settings.chunk_overlap_chars),
        )
        if not chunks:
            raise IngestionError("The uploaded document did not contain any extractable text.")

        version.parser_name = parsed_document.parser_name
        version.parser_version = parsed_document.parser_version
        version.metadata_ = {
            **(version.metadata_ or {}),
            **parsed_document.metadata,
            "chunk_count": len(chunks),
        }

        await index_document_chunks(
            session=session,
            settings=settings,
            document=document,
            version=version,
            stored_file=stored_file,
            chunks=chunks,
        )

        document.status = DocumentStatus.READY
        document.metadata_ = {
            **(document.metadata_ or {}),
            "chunk_count": len(chunks),
            "parser_name": parsed_document.parser_name,
            "parser_version": parsed_document.parser_version,
        }
        version.status = DocumentVersionStatus.READY
        await session.commit()
    except Exception as exc:
        document.status = DocumentStatus.FAILED
        version.status = DocumentVersionStatus.FAILED
        document.metadata_ = {**(document.metadata_ or {}), "error_message": str(exc)}
        version.metadata_ = {**(version.metadata_ or {}), "error_message": str(exc)}
        await session.commit()
        raise IngestionError(str(exc)) from exc
    finally:
        object_store.cleanup_staging_file(stored_file)

    refreshed = await get_document(session=session, document_id=document.id)
    return refreshed or document


async def list_documents(*, session: AsyncSession, limit: int = 50, offset: int = 0) -> list[Document]:
    statement = (
        select(Document)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
        .options(selectinload(Document.versions), selectinload(Document.qdrant_chunk_indexes))
    )
    result = await session.execute(statement)
    return list(result.scalars().unique().all())


async def get_document(*, session: AsyncSession, document_id: uuid.UUID) -> Document | None:
    statement = (
        select(Document)
        .where(Document.id == document_id)
        .options(selectinload(Document.versions), selectinload(Document.qdrant_chunk_indexes))
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


def _validate_supported_upload(upload: UploadFile) -> None:
    filename = upload.filename or "document"
    if not is_supported_document(filename, content_type=upload.content_type):
        raise UnsupportedDocumentTypeError("Unsupported document type. Supported formats are PDF, text, and markdown.")


def _create_document(*, stored_file: StoredDocumentFile, title: str | None) -> Document:
    return Document(
        title=(title or Path(stored_file.original_filename).stem or "Untitled document").strip(),
        original_filename=stored_file.original_filename,
        content_type=stored_file.content_type,
        storage_uri=stored_file.storage_uri,
        size_bytes=stored_file.size_bytes,
        checksum_sha256=stored_file.checksum_sha256,
        status=DocumentStatus.PROCESSING,
        metadata_=_storage_metadata(stored_file),
    )


def _storage_metadata(stored_file: StoredDocumentFile) -> dict[str, str | None]:
    return {
        "storage_backend": stored_file.storage_backend,
        "bucket_name": stored_file.bucket_name,
        "object_key": stored_file.object_key,
        "storage_uri": stored_file.storage_uri,
    }


async def _next_version_number(session: AsyncSession, document_id: uuid.UUID) -> int:
    statement = select(func.coalesce(func.max(DocumentVersion.version_number), 0)).where(
        DocumentVersion.document_id == document_id,
    )
    result = await session.execute(statement)
    return int(result.scalar_one()) + 1
