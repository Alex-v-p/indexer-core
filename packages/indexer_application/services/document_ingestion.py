from __future__ import annotations

import uuid
from pathlib import Path

from packages.indexer_application.dto import DocumentIngestionConfig, DocumentRecord
from packages.indexer_application.ports import (
    CacheInvalidator,
    DocumentObjectStore,
    DocumentStorageError,
    UnitOfWork,
    UploadFile,
)
from packages.indexer_application.services.chunk_indexing import index_document_chunks
from packages.rag_core.documents import (
    ChunkingConfig,
    UnsupportedDocumentTypeError,
    chunk_document,
    is_supported_document,
    parse_document,
)
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter


class IngestionError(RuntimeError):
    """Raised when an upload could not be ingested."""


async def ingest_uploaded_document(
    *,
    uow: UnitOfWork,
    config: DocumentIngestionConfig,
    upload: UploadFile,
    object_store: DocumentObjectStore,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndexWriter,
    keyword_cache: CacheInvalidator,
    title: str | None = None,
) -> DocumentRecord:
    """Store, parse, chunk, embed, and index one uploaded document."""

    _validate_supported_upload(upload)
    try:
        stored_file = await object_store.save_upload(upload)
    except DocumentStorageError as exc:
        raise IngestionError(str(exc)) from exc

    resolved_title = (title or Path(stored_file.original_filename).stem or "Untitled document").strip()
    document_id = await uow.documents.create_processing_document(stored_file=stored_file, title=resolved_title)
    version_id = await uow.documents.create_processing_version(document_id=document_id, stored_file=stored_file)

    try:
        parsed_document = parse_document(
            stored_file.path,
            filename=stored_file.original_filename,
            content_type=stored_file.content_type,
        )
        chunks = chunk_document(
            parsed_document,
            config=ChunkingConfig(
                max_chars=config.chunk_max_chars,
                overlap_chars=config.chunk_overlap_chars,
            ),
        )
        if not chunks:
            raise IngestionError("The uploaded document did not contain any extractable text.")

        await uow.documents.set_version_parser_metadata(
            version_id=version_id,
            parser_name=parsed_document.parser_name,
            parser_version=parsed_document.parser_version,
            metadata={**parsed_document.metadata, "chunk_count": len(chunks)},
        )
        await index_document_chunks(
            uow=uow,
            config=config,
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            keyword_cache=keyword_cache,
            document_id=document_id,
            version_id=version_id,
            stored_file=stored_file,
            chunks=chunks,
        )
        await uow.documents.mark_ready(
            document_id=document_id,
            version_id=version_id,
            document_metadata={
                "chunk_count": len(chunks),
                "parser_name": parsed_document.parser_name,
                "parser_version": parsed_document.parser_version,
            },
        )
        await uow.commit()
    except Exception as exc:
        await uow.documents.mark_failed(
            document_id=document_id,
            version_id=version_id,
            error_message=str(exc),
        )
        await uow.commit()
        raise IngestionError(str(exc)) from exc
    finally:
        object_store.cleanup_staging_file(stored_file)

    document = await uow.documents.get(document_id)
    if document is None:
        raise IngestionError("The ingested document could not be reloaded.")
    return document


async def list_documents(*, uow: UnitOfWork, limit: int = 50, offset: int = 0) -> list[DocumentRecord]:
    return await uow.documents.list(limit=limit, offset=offset)


async def get_document(*, uow: UnitOfWork, document_id: uuid.UUID) -> DocumentRecord | None:
    return await uow.documents.get(document_id)


def _validate_supported_upload(upload: UploadFile) -> None:
    filename = upload.filename or "document"
    if not is_supported_document(filename, content_type=upload.content_type):
        raise UnsupportedDocumentTypeError("Unsupported document type. Supported formats are PDF, text, and markdown.")
