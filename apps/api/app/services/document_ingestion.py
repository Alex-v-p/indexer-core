from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.database.models import (
    Document,
    DocumentStatus,
    DocumentVersion,
    DocumentVersionStatus,
    QdrantChunkIndex,
)
from app.core.config import Settings
from app.services.document_storage import StoredDocumentFile, build_document_storage
from packages.rag_core.documents import (
    ChunkingConfig,
    UnsupportedDocumentTypeError,
    chunk_document,
    is_supported_document,
    parse_document,
)
from packages.rag_core.documents.models import DocumentChunk
from packages.rag_core.providers import HashingEmbeddingProvider, QdrantVectorStore, VectorPoint


class VectorStore(Protocol):
    async def ensure_collection(self) -> None:
        """Ensure the target collection exists."""

    async def upsert_points(self, points: list[VectorPoint], *, batch_size: int = 64) -> None:
        """Upsert vector points."""


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

    filename = upload.filename or "document"
    if not is_supported_document(filename, content_type=upload.content_type):
        raise UnsupportedDocumentTypeError("Unsupported document type. Supported formats are PDF, text, and markdown.")

    stored_file = await build_document_storage(settings).save_upload(upload)
    if settings.max_upload_size_mb > 0 and stored_file.size_bytes > settings.max_upload_size_mb * 1024 * 1024:
        raise IngestionError(f"Uploaded file exceeds the {settings.max_upload_size_mb} MB limit.")

    document = Document(
        title=(title or Path(stored_file.original_filename).stem or "Untitled document").strip(),
        original_filename=stored_file.original_filename,
        content_type=stored_file.content_type,
        storage_uri=stored_file.storage_uri,
        size_bytes=stored_file.size_bytes,
        checksum_sha256=stored_file.checksum_sha256,
        status=DocumentStatus.PROCESSING,
        metadata_={"storage_backend": "local", "storage_path": str(stored_file.path)},
    )
    session.add(document)
    await session.flush()

    version = DocumentVersion(
        document_id=document.id,
        version_number=await _next_version_number(session, document.id),
        storage_uri=stored_file.storage_uri,
        content_type=stored_file.content_type,
        checksum_sha256=stored_file.checksum_sha256,
        status=DocumentVersionStatus.PROCESSING,
        metadata_={"storage_path": str(stored_file.path)},
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

        await _index_chunks(
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


async def _index_chunks(
    *,
    session: AsyncSession,
    settings: Settings,
    document: Document,
    version: DocumentVersion,
    stored_file: StoredDocumentFile,
    chunks: list[DocumentChunk],
) -> None:
    embedding_provider = HashingEmbeddingProvider(vector_size=settings.embedding_vector_size)
    vector_store = QdrantVectorStore(
        base_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_vector_size,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )

    await vector_store.ensure_collection()
    embeddings = await embedding_provider.embed_texts([chunk.text for chunk in chunks])

    points: list[VectorPoint] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_index_id = uuid.uuid4()
        point_id = str(uuid.uuid4())
        metadata = {
            **chunk.metadata,
            "document_id": str(document.id),
            "document_version_id": str(version.id),
            "qdrant_chunk_index_id": str(chunk_index_id),
            "original_filename": stored_file.original_filename,
            "storage_uri": stored_file.storage_uri,
        }
        session.add(
            QdrantChunkIndex(
                id=chunk_index_id,
                document_id=document.id,
                document_version_id=version.id,
                ordinal=chunk.ordinal,
                content_hash=chunk.content_hash,
                token_count=chunk.token_count,
                source_page_start=chunk.source_page_start,
                source_page_end=chunk.source_page_end,
                section_title=chunk.section_title,
                qdrant_collection=settings.qdrant_collection,
                qdrant_point_id=point_id,
                metadata_=metadata,
            ),
        )
        points.append(
            VectorPoint(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "content_hash": chunk.content_hash,
                    "token_count": chunk.token_count,
                    **metadata,
                },
            ),
        )

    await session.flush()
    await vector_store.upsert_points(points)


async def _next_version_number(session: AsyncSession, document_id: uuid.UUID) -> int:
    statement = select(func.coalesce(func.max(DocumentVersion.version_number), 0)).where(
        DocumentVersion.document_id == document_id,
    )
    result = await session.execute(statement)
    return int(result.scalar_one()) + 1
