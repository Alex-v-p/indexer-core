from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime

from packages.indexer_application.dto import DocumentIngestionConfig, DocumentRecord
from packages.indexer_application.ports import (
    CacheInvalidator,
    DocumentObjectStore,
    DocumentStorageError,
    DocumentVersionIndexActivator,
    MaterializedDocumentFile,
    UnitOfWork,
    UploadFile,
)
from packages.indexer_application.services.ingestion.activate import (
    ActivateDocumentInput,
    activate_document,
)
from packages.indexer_application.services.ingestion.contextualize import (
    ContextualizationInput,
    contextualize_document,
)
from packages.indexer_application.services.ingestion.errors import IngestionError
from packages.indexer_application.services.ingestion.index import IndexDocumentInput, index_document
from packages.indexer_application.services.ingestion.parse import ParseDocumentInput, parse_document_content
from packages.indexer_application.services.ingestion.prepare import (
    PrepareDocumentInput,
    PreparedDocument,
    prepare_document,
)
from packages.rag_core.documents import UnsupportedDocumentTypeError, is_supported_document
from packages.rag_core.ingestion import ChunkContextualizer, DocumentContextHierarchyBuilder
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    upload: UploadFile
    title: str | None = None
    version_of_document_id: uuid.UUID | None = None
    detect_existing_versions: bool = True
    published_at: date | datetime | None = None


class DocumentIngestionCoordinator:
    """Own synchronous ingestion ordering, failures, commits, and temporary cleanup."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        config: DocumentIngestionConfig,
        object_store: DocumentObjectStore,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndexWriter,
        version_index: DocumentVersionIndexActivator,
        keyword_cache: CacheInvalidator,
        contextualizer: ChunkContextualizer | None = None,
        hierarchy_builder: DocumentContextHierarchyBuilder | None = None,
    ) -> None:
        self._uow = uow
        self._config = config
        self._object_store = object_store
        self._embedding_provider = embedding_provider
        self._vector_index = vector_index
        self._version_index = version_index
        self._keyword_cache = keyword_cache
        self._contextualizer = contextualizer
        self._hierarchy_builder = hierarchy_builder

    async def ingest(self, request: IngestionRequest) -> DocumentRecord:
        _validate_supported_upload(request.upload)
        try:
            stored_document = await self._object_store.save_upload(request.upload)
        except DocumentStorageError as exc:
            raise IngestionError(str(exc)) from exc

        prepared: PreparedDocument | None = None
        materialized: MaterializedDocumentFile | None = None
        try:
            prepared = await prepare_document(
                uow=self._uow,
                request=PrepareDocumentInput(
                    stored_document=stored_document,
                    title=request.title,
                    version_of_document_id=request.version_of_document_id,
                    detect_existing_versions=request.detect_existing_versions,
                    published_at=request.published_at,
                ),
            )
            try:
                materialized = await self._object_store.materialize(stored_document)
            except DocumentStorageError as exc:
                raise IngestionError(str(exc)) from exc

            parsed = await parse_document_content(
                request=ParseDocumentInput(
                    materialized_document=materialized,
                    config=self._config,
                ),
                embedding_provider=self._embedding_provider,
            )
            contextualized = await contextualize_document(
                request=ContextualizationInput(config=self._config, parsed=parsed),
                contextualizer=self._contextualizer,
                hierarchy_builder=self._hierarchy_builder,
            )
            indexed = await index_document(
                request=IndexDocumentInput(
                    config=self._config,
                    prepared=prepared,
                    parsed=parsed,
                    contextualized=contextualized,
                ),
                uow=self._uow,
                embedding_provider=self._embedding_provider,
                vector_index=self._vector_index,
                keyword_cache=self._keyword_cache,
            )
            await activate_document(
                request=ActivateDocumentInput(
                    prepared=prepared,
                    parsed=parsed,
                    contextualized=contextualized,
                    indexed=indexed,
                ),
                uow=self._uow,
                version_index=self._version_index,
            )
            await self._uow.commit()
        except Exception as exc:
            if prepared is not None:
                await self._uow.documents.mark_failed(
                    document_id=prepared.document_id,
                    version_id=prepared.version.id,
                    error_message=str(exc),
                )
                await self._uow.commit()
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(str(exc)) from exc
        finally:
            if materialized is not None:
                self._object_store.cleanup_materialized_file(materialized)

        document = await self._uow.documents.get(prepared.document_id)
        if document is None:
            raise IngestionError("The ingested document could not be reloaded.")
        return document


def _validate_supported_upload(upload: UploadFile) -> None:
    filename = upload.filename or "document"
    if not is_supported_document(filename, content_type=upload.content_type):
        raise UnsupportedDocumentTypeError(
            "Unsupported document type. Supported formats are PDF, text, and markdown."
        )
