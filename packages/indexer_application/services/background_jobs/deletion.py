from __future__ import annotations

import uuid

from packages.indexer_application.dto import DocumentRecord
from packages.indexer_application.ports import (
    CacheInvalidator,
    DocumentIndexCleaner,
    DocumentObjectStore,
    StoredDocumentReference,
    UnitOfWork,
)
from packages.indexer_application.services.background_jobs.execution import ProgressReporter
from packages.indexer_application.services.background_jobs.payloads import stored_document_from_record


class DeleteDocumentJobHandler:
    """Idempotently remove one document from every owned persistence system."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        object_store: DocumentObjectStore,
        document_index: DocumentIndexCleaner,
        keyword_cache: CacheInvalidator,
    ) -> None:
        self._uow = uow
        self._object_store = object_store
        self._document_index = document_index
        self._keyword_cache = keyword_cache

    async def __call__(
        self,
        payload: dict[str, object],
        report: ProgressReporter,
    ) -> dict[str, object]:
        document_id = uuid.UUID(_required_string(payload, "document_id"))
        await report(0.05, "loading_document_for_deletion")
        document = await self._uow.documents.get(document_id)
        if document is None:
            return {
                "document_id": str(document_id),
                "already_deleted": True,
                "deleted_object_count": 0,
            }

        references = _stored_references(document)
        await report(0.20, "removing_document_vectors")
        await self._document_index.delete_document(document_id=document_id)

        deleted_objects = 0
        total_objects = max(len(references), 1)
        for index, reference in enumerate(references, start=1):
            progress = 0.30 + (0.40 * index / total_objects)
            await report(progress, "removing_stored_source_files")
            await self._object_store.delete(reference)
            deleted_objects += 1

        await report(0.78, "invalidating_keyword_index")
        self._keyword_cache.invalidate()

        await report(0.88, "removing_document_records")
        deleted = await self._uow.documents.delete(document_id=document_id)
        if not deleted:
            return {
                "document_id": str(document_id),
                "already_deleted": True,
                "deleted_object_count": deleted_objects,
            }

        await report(0.96, "document_deletion_complete")
        return {
            "document_id": str(document_id),
            "document_title": document.title,
            "deleted_version_count": len(document.versions),
            "deleted_chunk_registry_count": len(document.chunk_indexes),
            "deleted_object_count": deleted_objects,
            "already_deleted": False,
        }


def _stored_references(document: DocumentRecord) -> tuple[StoredDocumentReference, ...]:
    unique: dict[str, StoredDocumentReference] = {}
    for version in document.versions:
        reference = stored_document_from_record(document=document, version=version)
        unique.setdefault(reference.storage_uri, reference)
    return tuple(unique.values())


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()
