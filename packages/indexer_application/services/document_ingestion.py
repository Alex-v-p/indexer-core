from __future__ import annotations

import uuid
from datetime import date, datetime

from packages.indexer_application.dto import DocumentIngestionConfig, DocumentRecord
from packages.indexer_application.ports import (
    CacheInvalidator,
    DocumentObjectStore,
    DocumentVersionIndexActivator,
    UnitOfWork,
    UploadFile,
)
from packages.indexer_application.queries import (
    GetDocumentHandler,
    GetDocumentQuery,
    ListDocumentsHandler,
    ListDocumentsQuery,
)
from packages.indexer_application.services.ingestion import (
    DocumentIngestionCoordinator,
    IngestionError,
    IngestionRequest,
)
from packages.rag_core.ingestion import ChunkContextualizer, DocumentContextHierarchyBuilder
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter


class _VectorIndexActivationCompatibilityAdapter:
    """Preserve the legacy facade while composition injects the explicit port."""

    def __init__(self, vector_index: VectorIndexWriter) -> None:
        self._vector_index = vector_index

    async def activate_document_version(self, *, document_id: uuid.UUID, version_id: uuid.UUID) -> None:
        await self._vector_index.mark_document_version_current(
            document_id=str(document_id),
            version_id=str(version_id),
        )


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
    version_index: DocumentVersionIndexActivator | None = None,
) -> DocumentRecord:
    """Compatibility facade over the synchronous ingestion coordinator."""

    coordinator = DocumentIngestionCoordinator(
        uow=uow,
        config=config,
        object_store=object_store,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        version_index=version_index or _VectorIndexActivationCompatibilityAdapter(vector_index),
        keyword_cache=keyword_cache,
        contextualizer=contextualizer,
        hierarchy_builder=hierarchy_builder,
    )
    return await coordinator.ingest(
        IngestionRequest(
            upload=upload,
            title=title,
            version_of_document_id=version_of_document_id,
            detect_existing_versions=detect_existing_versions,
            published_at=published_at,
        )
    )


async def list_documents(*, uow: UnitOfWork, limit: int = 50, offset: int = 0) -> list[DocumentRecord]:
    """Compatibility facade; new callers should use ``ListDocumentsHandler``."""

    return await ListDocumentsHandler(uow=uow)(
        ListDocumentsQuery(limit=limit, offset=offset),
    )


async def get_document(*, uow: UnitOfWork, document_id: uuid.UUID) -> DocumentRecord | None:
    """Compatibility facade; new callers should use ``GetDocumentHandler``."""

    return await GetDocumentHandler(uow=uow)(GetDocumentQuery(document_id=document_id))


__all__ = ["IngestionError", "get_document", "ingest_uploaded_document", "list_documents"]
