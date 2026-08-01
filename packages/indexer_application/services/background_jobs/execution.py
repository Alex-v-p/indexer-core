from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Awaitable, Callable

from packages.indexer_application.dto import DocumentIngestionConfig, DocumentVersionIdentity
from packages.indexer_application.commands.classify_document_subjects import (
    enqueue_document_subject_classification,
)
from packages.indexer_application.ports import (
    CacheInvalidator,
    DocumentObjectStore,
    DocumentStorageError,
    DocumentVersionIndexActivator,
    UnitOfWork,
)
from packages.indexer_application.services.background_jobs.payloads import (
    prepared_document_from_payload,
    stored_document_from_record,
)
from packages.indexer_application.services.chunk_indexing import chunk_point_id
from packages.indexer_application.services.ingestion.activate import ActivateDocumentInput, activate_document
from packages.indexer_application.services.ingestion.contextualize import (
    ContextualizationInput,
    contextualize_document,
)
from packages.indexer_application.services.ingestion.errors import IngestionError
from packages.indexer_application.services.ingestion.index import IndexDocumentInput, index_document
from packages.indexer_application.services.ingestion.parse import ParseDocumentInput, parse_document_content
from packages.indexer_application.services.ingestion.prepare import PreparedDocument
from packages.indexer_application.services.hierarchy_indexing import (
    hierarchy_document_point_id,
    hierarchy_section_point_id,
)
from packages.rag_core.ingestion import ChunkContextualizer, DocumentContextHierarchyBuilder
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter

ProgressReporter = Callable[[float, str], Awaitable[None]]


class ProcessDocumentIngestionJobHandler:
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
        subject_classification_enabled: bool = False,
        subject_classification_policy_version: str = "subject-decision-policy/1.0",
        subject_classification_max_attempts: int = 3,
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
        self._subject_classification_enabled = subject_classification_enabled
        self._subject_classification_policy_version = subject_classification_policy_version
        self._subject_classification_max_attempts = subject_classification_max_attempts

    async def __call__(self, payload: dict[str, object], report: ProgressReporter) -> dict[str, object]:
        prepared = prepared_document_from_payload(payload)
        await report(0.05, "materializing_document")
        materialized = None
        try:
            try:
                materialized = await self._object_store.materialize(prepared.stored_document)
            except DocumentStorageError as exc:
                raise IngestionError(str(exc)) from exc
            parsed = await _parse(
                materialized=materialized,
                config=self._config,
                embedding_provider=self._embedding_provider,
                report=report,
            )
            contextualized = await _contextualize(
                parsed=parsed,
                config=self._config,
                contextualizer=self._contextualizer,
                hierarchy_builder=self._hierarchy_builder,
                report=report,
            )
            await report(0.72, "indexing_document")
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
            await report(0.92, "activating_document_version")
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
            classification_job = None
            if self._subject_classification_enabled:
                classification_job = await enqueue_document_subject_classification(
                    uow=self._uow,
                    document_id=prepared.document_id,
                    version=prepared.version,
                    policy_version=self._subject_classification_policy_version,
                    max_attempts=self._subject_classification_max_attempts,
                )
            return {
                "document_id": str(prepared.document_id),
                "document_version_id": str(prepared.version.id),
                "document_version_number": prepared.version.version_number,
                "chunk_count": len(parsed.chunks),
                "contextualization_status": contextualized.contextualization_metadata.get("status"),
                "hierarchical_indexing_status": indexed.hierarchical_retrieval_metadata.get("status"),
                "subject_classification_job_id": (
                    str(classification_job.id) if classification_job is not None else None
                ),
            }
        finally:
            if materialized is not None:
                self._object_store.cleanup_materialized_file(materialized)


class ReindexDocumentJobHandler:
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
        subject_classification_enabled: bool = False,
        subject_classification_policy_version: str = "subject-decision-policy/1.0",
        subject_classification_max_attempts: int = 3,
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
        self._subject_classification_enabled = subject_classification_enabled
        self._subject_classification_policy_version = subject_classification_policy_version
        self._subject_classification_max_attempts = subject_classification_max_attempts

    async def __call__(
        self,
        payload: dict[str, object],
        report: ProgressReporter,
        *,
        require_contextualization: bool = False,
    ) -> dict[str, object]:
        document_id = uuid.UUID(_required_string(payload, "document_id"))
        version_id = uuid.UUID(_required_string(payload, "document_version_id"))
        document = await self._uow.documents.get(document_id)
        if document is None:
            raise IngestionError(f"Document {document_id} was not found.")
        version = next((item for item in document.versions if item.id == version_id), None)
        if version is None:
            raise IngestionError(f"Document version {version_id} was not found.")

        ready_versions = [item for item in document.versions if item.status.value == "ready"]
        latest_ready = max(ready_versions, key=lambda item: item.version_number, default=None)
        if latest_ready is None or latest_ready.id != version_id:
            raise IngestionError("Only the latest ready document version can be rebuilt in place.")

        old_hierarchy_point_ids = _hierarchy_point_ids(version)
        config = replace(self._config, contextualization_enabled=True) if require_contextualization else self._config
        if require_contextualization and self._contextualizer is None:
            raise IngestionError("Contextualization is not configured for this worker.")

        stored_document = stored_document_from_record(document=document, version=version)
        prepared = PreparedDocument(
            stored_document=stored_document,
            document_id=document.id,
            document_title=document.title,
            version=DocumentVersionIdentity(
                id=version.id,
                version_number=version.version_number,
                uploaded_at=version.created_at,
                published_at=version.published_at,
            ),
            version_detection_method="background_reindex",
            matched_existing_document=True,
        )

        await report(0.05, "materializing_document")
        materialized = None
        try:
            try:
                materialized = await self._object_store.materialize(stored_document)
            except DocumentStorageError as exc:
                raise IngestionError(str(exc)) from exc
            parsed = await _parse(
                materialized=materialized,
                config=config,
                embedding_provider=self._embedding_provider,
                report=report,
            )
            contextualized = await _contextualize(
                parsed=parsed,
                config=config,
                contextualizer=self._contextualizer,
                hierarchy_builder=self._hierarchy_builder,
                report=report,
            )
            await report(0.68, "replacing_chunk_registry")
            old_point_ids = await self._uow.documents.delete_chunk_indexes(version_id=version.id)
            indexed = await index_document(
                request=IndexDocumentInput(
                    config=config,
                    prepared=prepared,
                    parsed=parsed,
                    contextualized=contextualized,
                ),
                uow=self._uow,
                embedding_provider=self._embedding_provider,
                vector_index=self._vector_index,
                keyword_cache=self._keyword_cache,
            )
            new_chunk_point_ids = {
                chunk_point_id(version.id, chunk.ordinal) for chunk in parsed.chunks
            }
            stale_chunk_point_ids = set(old_point_ids) - new_chunk_point_ids
            new_hierarchy_point_ids = _hierarchy_point_ids_from_metadata(
                version_id=version.id,
                metadata=indexed.hierarchical_retrieval_metadata,
            )
            stale_hierarchy_point_ids = old_hierarchy_point_ids - new_hierarchy_point_ids
            superseded_point_ids = sorted(stale_chunk_point_ids | stale_hierarchy_point_ids)
            await report(0.88, "removing_superseded_points")
            await self._vector_index.delete_points(superseded_point_ids)
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
            classification_job = None
            if self._subject_classification_enabled:
                classification_job = await enqueue_document_subject_classification(
                    uow=self._uow,
                    document_id=document.id,
                    version=version,
                    policy_version=self._subject_classification_policy_version,
                    max_attempts=self._subject_classification_max_attempts,
                )
            return {
                "document_id": str(document.id),
                "document_version_id": str(version.id),
                "document_version_number": version.version_number,
                "chunk_count": len(parsed.chunks),
                "replaced_point_count": len(old_point_ids),
                "contextualization_status": contextualized.contextualization_metadata.get("status"),
                "operation": "contextualization" if require_contextualization else "index_rebuild",
                "subject_classification_job_id": (
                    str(classification_job.id) if classification_job is not None else None
                ),
            }
        finally:
            if materialized is not None:
                self._object_store.cleanup_materialized_file(materialized)


async def _parse(
    *,
    materialized,
    config: DocumentIngestionConfig,
    embedding_provider: EmbeddingProvider,
    report: ProgressReporter,
):
    await report(0.12, "parsing_and_chunking")
    parsed = await parse_document_content(
        request=ParseDocumentInput(materialized_document=materialized, config=config),
        embedding_provider=embedding_provider,
    )
    await report(0.36, "building_context_hierarchy")
    return parsed


async def _contextualize(
    *,
    parsed,
    config: DocumentIngestionConfig,
    contextualizer: ChunkContextualizer | None,
    hierarchy_builder: DocumentContextHierarchyBuilder | None,
    report: ProgressReporter,
):
    contextualized = await contextualize_document(
        request=ContextualizationInput(config=config, parsed=parsed),
        contextualizer=contextualizer,
        hierarchy_builder=hierarchy_builder,
    )
    await report(0.64, "contextualization_complete")
    return contextualized


def _hierarchy_point_ids(version) -> set[str]:
    hierarchy = version.metadata.get("hierarchical_retrieval")
    if not isinstance(hierarchy, dict):
        return set()
    return _hierarchy_point_ids_from_metadata(version_id=version.id, metadata=hierarchy)


def _hierarchy_point_ids_from_metadata(
    *,
    version_id: uuid.UUID,
    metadata: dict[str, object],
) -> set[str]:
    if metadata.get("enabled") is not True or metadata.get("status") != "ready":
        return set()
    sections = metadata.get("sections")
    if not isinstance(sections, list):
        sections = []
    point_ids: set[str] = {hierarchy_document_point_id(version_id)}
    for section in sections:
        if not isinstance(section, dict):
            continue
        cluster_id = section.get("cluster_id")
        if isinstance(cluster_id, int) and not isinstance(cluster_id, bool) and cluster_id > 0:
            point_ids.add(hierarchy_section_point_id(version_id, cluster_id))
    return point_ids


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()
