from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path

from packages.indexer_application.dto import DocumentRecord, DocumentVersionIdentity
from packages.indexer_application.ports import StoredDocumentReference, UnitOfWork
from packages.indexer_application.services.ingestion.errors import IngestionError
from packages.rag_core.documents.version_families import normalized_document_identity


@dataclass(frozen=True, slots=True)
class PrepareDocumentInput:
    stored_document: StoredDocumentReference
    title: str | None = None
    version_of_document_id: uuid.UUID | None = None
    detect_existing_versions: bool = True
    published_at: date | datetime | None = None


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    stored_document: StoredDocumentReference
    document_id: uuid.UUID
    document_title: str
    version: DocumentVersionIdentity
    version_detection_method: str
    matched_existing_document: bool


async def prepare_document(
    *,
    uow: UnitOfWork,
    request: PrepareDocumentInput,
) -> PreparedDocument:
    """Resolve document/version identity and create the processing records."""

    stored_document = request.stored_document
    resolved_title = (
        request.title or Path(stored_document.original_filename).stem or "Untitled document"
    ).strip()
    existing_document: DocumentRecord | None = None
    version_detection_method = "new_document"

    if request.version_of_document_id is not None:
        existing_document = await uow.documents.get(request.version_of_document_id)
        if existing_document is None:
            raise IngestionError(f"Document {request.version_of_document_id} was not found.")
        version_detection_method = "explicit_document_id"
    elif request.detect_existing_versions:
        existing_document = await uow.documents.find_version_candidate(
            title=resolved_title,
            original_filename=stored_document.original_filename,
        )
        if existing_document is not None:
            exact_identity_match = (
                normalized_document_identity(existing_document.title)
                == normalized_document_identity(resolved_title)
                or normalized_document_identity(existing_document.original_filename or "")
                == normalized_document_identity(stored_document.original_filename)
            )
            version_detection_method = (
                "matching_title_or_filename"
                if exact_identity_match
                else "matching_document_family"
            )

    if existing_document is None:
        document_id = await uow.documents.create_processing_document(
            stored_file=stored_document,
            title=resolved_title,
        )
        document_title = resolved_title
    else:
        document_id = existing_document.id
        document_title = existing_document.title

    create_version = uow.documents.create_processing_version
    create_version_kwargs = {
        "document_id": document_id,
        "stored_file": stored_document,
    }
    resolved_published_at = normalize_published_at(request.published_at)
    if "published_at" in inspect.signature(create_version).parameters:
        create_version_kwargs["published_at"] = resolved_published_at

    version = coerce_version_identity(await create_version(**create_version_kwargs))
    return PreparedDocument(
        stored_document=stored_document,
        document_id=document_id,
        document_title=document_title,
        version=version,
        version_detection_method=version_detection_method,
        matched_existing_document=existing_document is not None,
    )


def normalize_published_at(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return datetime.combine(value, time.min, tzinfo=UTC)


def coerce_version_identity(value: object) -> DocumentVersionIdentity:
    if isinstance(value, DocumentVersionIdentity):
        return value
    if isinstance(value, uuid.UUID):
        # Compatibility for older repository test doubles. Production repositories
        # return the real sequential version number.
        return DocumentVersionIdentity(id=value, version_number=1)
    raise TypeError("create_processing_version must return DocumentVersionIdentity.")
