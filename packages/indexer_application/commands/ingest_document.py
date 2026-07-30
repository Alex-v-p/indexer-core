from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime

from packages.indexer_application.dto import DocumentIngestionConfig, DocumentRecord
from packages.indexer_application.ports import (
    CacheInvalidator,
    DocumentObjectStore,
    UnitOfWork,
    UploadFile,
)
from packages.indexer_application.services.document_ingestion import (
    IngestionError,
    ingest_uploaded_document,
)
from packages.rag_core.ingestion import ChunkContextualizer, DocumentContextHierarchyBuilder
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter


@dataclass(frozen=True, slots=True)
class IngestDocumentCommand:
    upload: UploadFile
    title: str | None = None
    version_of_document_id: uuid.UUID | None = None
    detect_existing_versions: bool = True
    published_at: date | datetime | None = None


class IngestDocumentHandler:
    """Application entry point for the synchronous document-ingestion use case."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        config: DocumentIngestionConfig,
        object_store: DocumentObjectStore,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndexWriter,
        keyword_cache: CacheInvalidator,
        contextualizer: ChunkContextualizer | None = None,
        hierarchy_builder: DocumentContextHierarchyBuilder | None = None,
    ) -> None:
        self._uow = uow
        self._config = config
        self._object_store = object_store
        self._embedding_provider = embedding_provider
        self._vector_index = vector_index
        self._keyword_cache = keyword_cache
        self._contextualizer = contextualizer
        self._hierarchy_builder = hierarchy_builder

    async def __call__(self, command: IngestDocumentCommand) -> DocumentRecord:
        return await ingest_uploaded_document(
            uow=self._uow,
            config=self._config,
            upload=command.upload,
            object_store=self._object_store,
            embedding_provider=self._embedding_provider,
            vector_index=self._vector_index,
            keyword_cache=self._keyword_cache,
            contextualizer=self._contextualizer,
            hierarchy_builder=self._hierarchy_builder,
            title=command.title,
            version_of_document_id=command.version_of_document_id,
            detect_existing_versions=command.detect_existing_versions,
            published_at=command.published_at,
        )
