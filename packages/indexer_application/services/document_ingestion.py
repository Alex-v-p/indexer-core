from __future__ import annotations

import inspect
import logging
import uuid
from datetime import UTC, date, datetime, time
from pathlib import Path

from packages.indexer_application.dto import (
    DocumentIngestionConfig,
    DocumentRecord,
    DocumentVersionIdentity,
)
from packages.indexer_application.ports import (
    CacheInvalidator,
    DocumentObjectStore,
    DocumentStorageError,
    StoredDocumentFile,
    UnitOfWork,
    UploadFile,
)
from packages.indexer_application.queries import (
    GetDocumentHandler,
    GetDocumentQuery,
    ListDocumentsHandler,
    ListDocumentsQuery,
)
from packages.indexer_application.services.chunk_indexing import index_document_chunks
from packages.indexer_application.services.hierarchy_indexing import index_document_hierarchy
from packages.rag_core.documents import (
    ChunkingConfig,
    DocumentChunk,
    ParsedDocument,
    UnsupportedDocumentTypeError,
    chunk_document,
    is_supported_document,
    parse_document,
)
from packages.rag_core.documents.version_families import normalized_document_identity
from packages.rag_core.ingestion import (
    ChunkContextualizer,
    ContextualizedChunk,
    DocumentContextHierarchy,
    DocumentContextHierarchyBuilder,
)
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
    hierarchy_builder: DocumentContextHierarchyBuilder | None = None,
    title: str | None = None,
    version_of_document_id: uuid.UUID | None = None,
    detect_existing_versions: bool = True,
    published_at: date | datetime | None = None,
) -> DocumentRecord:
    """Store, parse, chunk, embed, optionally contextualize, and index a document."""

    _validate_supported_upload(upload)
    try:
        stored_file = await object_store.save_upload(upload)
    except DocumentStorageError as exc:
        raise IngestionError(str(exc)) from exc

    resolved_title = (title or Path(stored_file.original_filename).stem or "Untitled document").strip()
    existing_document = None
    version_detection_method = "new_document"
    if version_of_document_id is not None:
        existing_document = await uow.documents.get(version_of_document_id)
        if existing_document is None:
            object_store.cleanup_staging_file(stored_file)
            raise IngestionError(f"Document {version_of_document_id} was not found.")
        version_detection_method = "explicit_document_id"
    elif detect_existing_versions:
        finder = getattr(uow.documents, "find_version_candidate", None)
        if callable(finder):
            existing_document = await finder(
                title=resolved_title,
                original_filename=stored_file.original_filename,
            )
            if existing_document is not None:
                exact_identity_match = (
                    normalized_document_identity(existing_document.title)
                    == normalized_document_identity(resolved_title)
                    or normalized_document_identity(existing_document.original_filename or "")
                    == normalized_document_identity(stored_file.original_filename)
                )
                version_detection_method = (
                    "matching_title_or_filename"
                    if exact_identity_match
                    else "matching_document_family"
                )

    if existing_document is None:
        document_id = await uow.documents.create_processing_document(stored_file=stored_file, title=resolved_title)
        document_title = resolved_title
    else:
        document_id = existing_document.id
        document_title = existing_document.title

    resolved_published_at = _normalize_published_at(published_at)
    create_version = uow.documents.create_processing_version
    create_version_kwargs = {
        "document_id": document_id,
        "stored_file": stored_file,
    }
    if "published_at" in inspect.signature(create_version).parameters:
        create_version_kwargs["published_at"] = resolved_published_at
    version_identity = _coerce_version_identity(
        await create_version(**create_version_kwargs),
    )
    version_id = version_identity.id

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

        hierarchy, hierarchy_build_metadata = await _build_context_hierarchy(
            config=config,
            parsed_document=parsed_document,
            chunks=chunks,
            chunk_embeddings=original_embeddings,
            hierarchy_builder=hierarchy_builder,
            contextualizer=contextualizer,
        )
        contextualized_chunks, contextualization_metadata, contextualization_hierarchy = await _contextualize_chunks(
            config=config,
            parsed_document=parsed_document,
            chunks=chunks,
            chunk_embeddings=original_embeddings,
            contextualizer=contextualizer,
            hierarchy=hierarchy,
        )
        hierarchy = hierarchy or contextualization_hierarchy
        indexed_hierarchy = hierarchy if config.hierarchical_indexing_enabled else None
        uploaded_at = version_identity.uploaded_at or datetime.now(UTC)

        await index_document_chunks(
            uow=uow,
            config=config,
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            keyword_cache=keyword_cache,
            document_id=document_id,
            version_id=version_id,
            version_number=version_identity.version_number,
            uploaded_at=uploaded_at,
            published_at=version_identity.published_at,
            document_title=document_title,
            stored_file=stored_file,
            chunks=chunks,
            original_embeddings=original_embeddings,
            contextualized_chunks=contextualized_chunks,
            contextualization_metadata=contextualization_metadata,
            hierarchy=indexed_hierarchy,
            promote_version=False,
        )
        hierarchical_retrieval_metadata = await _index_hierarchy(
            config=config,
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            document_id=document_id,
            version_id=version_id,
            version_number=version_identity.version_number,
            uploaded_at=uploaded_at,
            published_at=version_identity.published_at,
            document_title=document_title,
            stored_file=stored_file,
            chunks=chunks,
            hierarchy=hierarchy,
            build_metadata=hierarchy_build_metadata,
        )

        await _promote_indexed_document_version(
            vector_index=vector_index,
            document_id=document_id,
            version_id=version_id,
        )

        await uow.documents.set_version_parser_metadata(
            version_id=version_id,
            parser_name=parsed_document.parser_name,
            parser_version=parsed_document.parser_version,
            metadata={
                **parsed_document.metadata,
                "chunk_count": len(chunks),
                "contextualization": contextualization_metadata,
                "hierarchical_retrieval": hierarchical_retrieval_metadata,
                "document_version_number": version_identity.version_number,
                "uploaded_at": _isoformat(version_identity.uploaded_at),
                "published_at": _isoformat(version_identity.published_at),
                "version_detection": {
                    "method": version_detection_method,
                    "matched_existing_document": existing_document is not None,
                },
            },
        )
        await uow.documents.mark_ready(
            document_id=document_id,
            version_id=version_id,
            document_metadata={
                "chunk_count": len(chunks),
                "parser_name": parsed_document.parser_name,
                "parser_version": parsed_document.parser_version,
                "contextualization": contextualization_metadata,
                "hierarchical_retrieval": hierarchical_retrieval_metadata,
                "latest_version_id": str(version_id),
                "latest_version_number": version_identity.version_number,
                "version_detection_method": version_detection_method,
            },
            stored_file=stored_file,
            version_number=version_identity.version_number,
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
    """Compatibility facade; new callers should use ``ListDocumentsHandler``."""

    return await ListDocumentsHandler(uow=uow)(
        ListDocumentsQuery(limit=limit, offset=offset),
    )


async def get_document(*, uow: UnitOfWork, document_id: uuid.UUID) -> DocumentRecord | None:
    """Compatibility facade; new callers should use ``GetDocumentHandler``."""

    return await GetDocumentHandler(uow=uow)(GetDocumentQuery(document_id=document_id))


def _normalize_published_at(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return datetime.combine(value, time.min, tzinfo=UTC)


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


async def _build_context_hierarchy(
    *,
    config: DocumentIngestionConfig,
    parsed_document: ParsedDocument,
    chunks: list[DocumentChunk],
    chunk_embeddings: list[list[float]],
    hierarchy_builder: DocumentContextHierarchyBuilder | None,
    contextualizer: ChunkContextualizer | None,
) -> tuple[DocumentContextHierarchy | None, dict[str, object]]:
    required = config.contextualization_enabled or config.hierarchical_indexing_enabled
    if not required:
        return None, {"enabled": False, "status": "disabled"}
    if hierarchy_builder is None:
        if contextualizer is not None and config.contextualization_enabled:
            return None, {"enabled": True, "status": "deferred_to_contextualizer"}
        if config.hierarchical_indexing_fail_open:
            return None, {
                "enabled": True,
                "status": "failed_open",
                "error": "No document context hierarchy builder is configured.",
            }
        raise IngestionError("Hierarchical indexing is enabled but no document context hierarchy builder is configured.")

    try:
        hierarchy = await hierarchy_builder.build(parsed_document, chunks, chunk_embeddings)
    except Exception as exc:
        hierarchy_can_fail_open = (
            (not config.hierarchical_indexing_enabled or config.hierarchical_indexing_fail_open)
            and (not config.contextualization_enabled or config.contextualization_fail_open)
        )
        if not hierarchy_can_fail_open:
            logger.exception("Document context hierarchy generation failed; aborting ingestion.")
            raise
        logger.warning("Document context hierarchy generation failed; continuing without hierarchy.", exc_info=True)
        return None, {"enabled": True, "status": "failed_open", "error": str(exc)}

    return hierarchy, {
        "enabled": True,
        "status": "ready",
        "strategy": "semantic_cluster_hierarchy",
        "cluster_count": len(hierarchy.clusters),
    }


async def _contextualize_chunks(
    *,
    config: DocumentIngestionConfig,
    parsed_document: ParsedDocument,
    chunks: list[DocumentChunk],
    chunk_embeddings: list[list[float]],
    contextualizer: ChunkContextualizer | None,
    hierarchy: DocumentContextHierarchy | None,
) -> tuple[list[ContextualizedChunk] | None, dict[str, object], DocumentContextHierarchy | None]:
    if not config.contextualization_enabled:
        logger.info(
            "Document contextualization is disabled; indexing original vectors only.",
            extra={"chunk_count": len(chunks)},
        )
        return None, {"enabled": False, "status": "disabled"}, hierarchy
    if contextualizer is None:
        raise IngestionError("Contextualization is enabled but no chunk contextualizer is configured.")

    representation_metadata = {
        "collection": config.vector_collection_name,
        "vector_name": config.contextual_vector_name,
    }
    logger.info(
        "Contextualizing chunks with the reusable document hierarchy before vector indexing.",
        extra={
            "document_title": parsed_document.title,
            "chunk_count": len(chunks),
            "contextual_vector_name": config.contextual_vector_name,
            "hierarchy_prebuilt": hierarchy is not None,
        },
    )
    try:
        contextualize = contextualizer.contextualize
        kwargs: dict[str, object] = {}
        if hierarchy is not None and "hierarchy" in inspect.signature(contextualize).parameters:
            kwargs["hierarchy"] = hierarchy
        contextualization_result = await contextualize(
            parsed_document,
            chunks,
            chunk_embeddings,
            **kwargs,
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
        }, hierarchy

    logger.info(
        "Document contextualization completed.",
        extra={"document_title": parsed_document.title, "chunk_count": len(contextualized_chunks)},
    )
    resolved_hierarchy = hierarchy or contextualization_result.hierarchy
    return contextualized_chunks, {
        "enabled": True,
        "status": "ready",
        "strategy": "semantic_cluster_hierarchy",
        "chunk_count": len(contextualized_chunks),
        "cluster_count": len(resolved_hierarchy.clusters),
        "document_summary": resolved_hierarchy.document_summary,
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "chunk_ordinals": list(cluster.chunk_ordinals),
                "summary": cluster.summary,
            }
            for cluster in resolved_hierarchy.clusters
        ],
        **representation_metadata,
    }, resolved_hierarchy


async def _index_hierarchy(
    *,
    config: DocumentIngestionConfig,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndexWriter,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    version_number: int,
    uploaded_at: datetime,
    published_at: datetime | None,
    document_title: str,
    stored_file: StoredDocumentFile,
    chunks: list[DocumentChunk],
    hierarchy: DocumentContextHierarchy | None,
    build_metadata: dict[str, object],
) -> dict[str, object]:
    if not config.hierarchical_indexing_enabled:
        return {"enabled": False, "status": "disabled"}
    metadata: dict[str, object] = {
        **build_metadata,
        "enabled": True,
        "collection": config.vector_collection_name,
        "vector_name": config.hierarchy_vector_name,
        "levels": ["document", "section", "chunk"],
    }
    if hierarchy is None:
        error = str(metadata.get("error") or "No document hierarchy was produced.")
        if not config.hierarchical_indexing_fail_open:
            raise IngestionError(error)
        return {**metadata, "status": "failed_open", "error": error}

    try:
        summary_point_count = await index_document_hierarchy(
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            hierarchy_vector_name=config.hierarchy_vector_name,
            document_id=document_id,
            version_id=version_id,
            version_number=version_number,
            uploaded_at=uploaded_at,
            published_at=published_at,
            document_title=document_title,
            stored_file=stored_file,
            chunks=chunks,
            hierarchy=hierarchy,
        )
    except Exception as exc:
        if not config.hierarchical_indexing_fail_open:
            logger.exception("Hierarchy summary indexing failed; aborting ingestion.")
            raise
        logger.warning("Hierarchy summary indexing failed; continuing with chunk indexes only.", exc_info=True)
        return {**metadata, "status": "failed_open", "error": str(exc)}

    return {
        **metadata,
        "status": "ready",
        "summary_point_count": summary_point_count,
        "document_summary_count": 1,
        "section_summary_count": len(hierarchy.clusters),
        "document_summary": hierarchy.document_summary,
        "sections": [
            {
                "cluster_id": cluster.cluster_id,
                "chunk_ordinals": list(cluster.chunk_ordinals),
                "summary": cluster.summary,
            }
            for cluster in hierarchy.clusters
        ],
    }


async def _promote_indexed_document_version(
    *,
    vector_index: VectorIndexWriter,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
) -> None:
    """Promote a version only after all required chunk and hierarchy points exist."""

    promote = getattr(vector_index, "mark_document_version_current", None)
    if callable(promote):
        await promote(document_id=str(document_id), version_id=str(version_id))


def _coerce_version_identity(value: object) -> DocumentVersionIdentity:
    if isinstance(value, DocumentVersionIdentity):
        return value
    if isinstance(value, uuid.UUID):
        # Compatibility for older repository test doubles. Production repositories
        # return the real sequential version number.
        return DocumentVersionIdentity(id=value, version_number=1)
    raise TypeError("create_processing_version must return DocumentVersionIdentity.")


def _validate_supported_upload(upload: UploadFile) -> None:
    filename = upload.filename or "document"
    if not is_supported_document(filename, content_type=upload.content_type):
        raise UnsupportedDocumentTypeError("Unsupported document type. Supported formats are PDF, text, and markdown.")
