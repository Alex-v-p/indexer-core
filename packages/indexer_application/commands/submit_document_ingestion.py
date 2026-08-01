from __future__ import annotations

import logging
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
from packages.indexer_application.ports import SubjectDecisionConflictError
from packages.indexer_application.services.background_jobs import prepared_document_to_payload
from packages.indexer_application.services.ingestion import IngestionError
from packages.indexer_application.services.ingestion.prepare import PrepareDocumentInput, prepare_document
from packages.rag_core.documents import UnsupportedDocumentTypeError, is_supported_document
from packages.rag_core.subjects import (
    DecisionControlSource,
    DecisionState,
    DocumentSubjectDecision,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SubmitDocumentIngestionCommand:
    upload: UploadFile
    title: str | None = None
    version_of_document_id: uuid.UUID | None = None
    detect_existing_versions: bool = True
    published_at: date | datetime | None = None
    subject_ids: tuple[uuid.UUID, ...] = ()


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
        subject_ids = await _validate_subject_ids(self._uow, command.subject_ids)
        try:
            stored_document = await self._object_store.save_upload(command.upload)
        except DocumentStorageError as exc:
            raise IngestionError(str(exc)) from exc

        try:
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
            await _apply_upload_subject_assignments(
                self._uow,
                document_id=prepared.document_id,
                subject_ids=subject_ids,
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
        except Exception:
            try:
                await self._uow.rollback()
            except Exception:
                logger.exception(
                    "Document ingestion rollback failed after upload submission failed.",
                )
            try:
                await self._object_store.delete(stored_document)
            except Exception:
                logger.exception(
                    "Could not remove the just-saved upload after document ingestion "
                    "submission failed: %s",
                    stored_document.storage_uri,
                )
            raise

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


async def _validate_subject_ids(
    uow: UnitOfWork,
    subject_ids: tuple[uuid.UUID, ...],
) -> tuple[uuid.UUID, ...]:
    unique_subject_ids = tuple(dict.fromkeys(subject_ids))
    for subject_id in unique_subject_ids:
        subject = await uow.subjects.get(subject_id, include_archived=True)
        if subject is None:
            raise IngestionError(f"Subject {subject_id} was not found.")
        if subject.archived_at is not None:
            raise IngestionError(f"Subject {subject_id} is archived and read-only.")
    return unique_subject_ids


async def _apply_upload_subject_assignments(
    uow: UnitOfWork,
    *,
    document_id: uuid.UUID,
    subject_ids: tuple[uuid.UUID, ...],
) -> None:
    for subject_id in subject_ids:
        current = await uow.subjects.get_decision(
            document_id=document_id,
            subject_id=subject_id,
        )
        if (
            current is not None
            and current.control_source is DecisionControlSource.MANUAL
        ):
            continue
        try:
            await uow.subjects.write_decision(
                DocumentSubjectDecision(
                    document_id=document_id,
                    subject_id=subject_id,
                    state=DecisionState.ASSIGNED,
                    control_source=DecisionControlSource.MANUAL,
                    rationale="Assigned during document upload.",
                ),
                expected_revision=current.revision if current is not None else 0,
            )
        except SubjectDecisionConflictError as exc:
            raise IngestionError(
                "A subject assignment changed while the document was being uploaded. "
                "Refresh the document and try again."
            ) from exc
        except LookupError as exc:
            raise IngestionError(
                "A selected subject was archived while the document was being uploaded. "
                "Refresh the subject list and try again."
            ) from exc
