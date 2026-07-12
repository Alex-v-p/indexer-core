from __future__ import annotations

import logging
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
    DocumentChunk,
    ParsedDocument,
    UnsupportedDocumentTypeError,
    chunk_document,
    is_supported_document,
    parse_document,
)
from packages.rag_core.ingestion import ChunkContextualizer, ContextualizedChunk
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter


logger = logging.getLogger(__name__)


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
    contextualizer: ChunkContextualizer | None = None,
    title: str | None = None,
) -> DocumentRecord:
    """Store, parse, chunk, embed, optionally contextualize, and index a document."""

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

        original_embeddings = await embedding_provider.embed_texts([chunk.text for chunk in chunks])
        if len(original_embeddings) != len(chunks):
            raise IngestionError("Embedding provider must return exactly one vector per source chunk.")

        contextualized_chunks, contextualization_metadata = await _contextualize_chunks(
            config=config,
            parsed_document=parsed_document,
            chunks=chunks,
            chunk_embeddings=original_embeddings,
            contextualizer=contextualizer,
        )

        await uow.documents.set_version_parser_metadata(
            version_id=version_id,
            parser_name=parsed_document.parser_name,
            parser_version=parsed_document.parser_version,
            metadata={
                **parsed_document.metadata,
                "chunk_count": len(chunks),
                "contextualization": contextualization_metadata,
            },
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
            original_embeddings=original_embeddings,
            contextualized_chunks=contextualized_chunks,
            contextualization_metadata=contextualization_metadata,
        )
        await uow.documents.mark_ready(
            document_id=document_id,
            version_id=version_id,
            document_metadata={
                "chunk_count": len(chunks),
                "parser_name": parsed_document.parser_name,
                "parser_version": parsed_document.parser_version,
                "contextualization": contextualization_metadata,
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


async def _contextualize_chunks(
    *,
    config: DocumentIngestionConfig,
    parsed_document: ParsedDocument,
    chunks: list[DocumentChunk],
    chunk_embeddings: list[list[float]],
    contextualizer: ChunkContextualizer | None,
) -> tuple[list[ContextualizedChunk] | None, dict[str, object]]:
    if not config.contextualization_enabled:
        logger.info(
            "Document contextualization is disabled; indexing original vectors only.",
            extra={"chunk_count": len(chunks)},
        )
        return None, {"enabled": False, "status": "disabled"}
    if contextualizer is None:
        raise IngestionError("Contextualization is enabled but no chunk contextualizer is configured.")

    representation_metadata = {
        "collection": config.vector_collection_name,
        "vector_name": config.contextual_vector_name,
    }
    logger.info(
        "Building semantic document context and contextualizing chunks before vector indexing.",
        extra={
            "document_title": parsed_document.title,
            "chunk_count": len(chunks),
            "contextual_vector_name": config.contextual_vector_name,
        },
    )
    try:
        contextualization_result = await contextualizer.contextualize(
            parsed_document,
            chunks,
            chunk_embeddings,
        )
        contextualized_chunks = contextualization_result.chunks
        if len(contextualized_chunks) != len(chunks):
            raise ValueError("Contextualizer must return exactly one representation per chunk.")
    except Exception as exc:
        if not config.contextualization_fail_open:
            logger.exception("Document contextualization failed; aborting ingestion.")
            raise
        logger.warning(
            "Document contextualization failed; continuing with original vectors only.",
            exc_info=True,
        )
        return None, {
            "enabled": True,
            "status": "failed_open",
            "error": str(exc),
            **representation_metadata,
        }

    logger.info(
        "Document contextualization completed.",
        extra={"document_title": parsed_document.title, "chunk_count": len(contextualized_chunks)},
    )
    hierarchy = contextualization_result.hierarchy
    return contextualized_chunks, {
        "enabled": True,
        "status": "ready",
        "strategy": "semantic_cluster_hierarchy",
        "chunk_count": len(contextualized_chunks),
        "cluster_count": len(hierarchy.clusters),
        "document_summary": hierarchy.document_summary,
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "chunk_ordinals": list(cluster.chunk_ordinals),
                "summary": cluster.summary,
            }
            for cluster in hierarchy.clusters
        ],
        **representation_metadata,
    }


def _validate_supported_upload(upload: UploadFile) -> None:
    filename = upload.filename or "document"
    if not is_supported_document(filename, content_type=upload.content_type):
        raise UnsupportedDocumentTypeError("Unsupported document type. Supported formats are PDF, text, and markdown.")
