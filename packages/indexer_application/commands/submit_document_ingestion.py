from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime

from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobSubmission,
    BackgroundJobType,
    DocumentRecord,
)
from packages.indexer_application.ports import DocumentObjectStore, DocumentStorageError, UnitOfWork, UploadFile
from packages.indexer_application.services.background_jobs import prepared_document_to_payload
from packages.indexer_application.services.ingestion import IngestionError
from packages.indexer_application.services.ingestion.prepare import PrepareDocumentInput, prepare_document
from packages.rag_core.documents import UnsupportedDocumentTypeError, is_supported_document


@dataclass(frozen=True, slots=True)
class SubmitDocumentIngestionCommand:
    upload: UploadFile
    title: str | None = None
    version_of_document_id: uuid.UUID | None = None
    detect_existing_versions: bool = True
    published_at: date | datetime | None = None


@dataclass(frozen=True, slots=True)
class QueuedDocumentIngestionResult:
    document: DocumentRecord
    job: BackgroundJobRecord


class SubmitDocumentIngestionHandler:
    """Persist an upload and enqueue the expensive ingestion phases."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        object_store: DocumentObjectStore,
        max_attempts: int = 3,
    ) -> None:
        self._uow = uow
        self._object_store = object_store
        self._max_attempts = max_attempts

    async def __call__(self, command: SubmitDocumentIngestionCommand) -> QueuedDocumentIngestionResult:
        _validate_supported_upload(command.upload)
        try:
            stored_document = await self._object_store.save_upload(command.upload)
        except DocumentStorageError as exc:
            raise IngestionError(str(exc)) from exc

        prepared = await prepare_document(
            uow=self._uow,
            request=PrepareDocumentInput(
                stored_document=stored_document,
                title=command.title,
                version_of_document_id=command.version_of_document_id,
                detect_existing_versions=command.detect_existing_versions,
                published_at=command.published_at,
            ),
        )
        job = await self._uow.background_jobs.enqueue(
            BackgroundJobSubmission(
                job_type=BackgroundJobType.INGEST_DOCUMENT,
                payload=prepared_document_to_payload(prepared),
                max_attempts=self._max_attempts,
                dedupe_key=f"ingestion:{prepared.version.id}",
            ),
        )
        await self._uow.commit()

        document = await self._uow.documents.get(prepared.document_id)
        if document is None:
            raise IngestionError("The queued document could not be reloaded.")
        return QueuedDocumentIngestionResult(document=document, job=job)


def _validate_supported_upload(upload: UploadFile) -> None:
    filename = upload.filename or "document"
    if not is_supported_document(filename, content_type=upload.content_type):
        raise UnsupportedDocumentTypeError(
            "Unsupported document type. Supported formats are PDF, text, and markdown.",
        )
