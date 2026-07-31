from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from packages.indexer_application.dto import DocumentRecord, DocumentVersionIdentity, DocumentVersionRecord
from packages.indexer_application.ports import StoredDocumentReference
from packages.indexer_application.services.ingestion.prepare import PreparedDocument


def prepared_document_to_payload(prepared: PreparedDocument) -> dict[str, Any]:
    return {
        "document_id": str(prepared.document_id),
        "document_title": prepared.document_title,
        "document_version_id": str(prepared.version.id),
        "document_version_number": prepared.version.version_number,
        "uploaded_at": _serialize_datetime(prepared.version.uploaded_at),
        "published_at": _serialize_datetime(prepared.version.published_at),
        "version_detection_method": prepared.version_detection_method,
        "matched_existing_document": prepared.matched_existing_document,
        "stored_document": _stored_document_to_payload(prepared.stored_document),
    }


def prepared_document_from_payload(payload: dict[str, Any]) -> PreparedDocument:
    stored_payload = _mapping(payload.get("stored_document"), "stored_document")
    return PreparedDocument(
        stored_document=_stored_document_from_payload(stored_payload),
        document_id=uuid.UUID(_required_string(payload, "document_id")),
        document_title=_required_string(payload, "document_title"),
        version=DocumentVersionIdentity(
            id=uuid.UUID(_required_string(payload, "document_version_id")),
            version_number=_positive_int(payload.get("document_version_number"), "document_version_number"),
            uploaded_at=_optional_datetime(payload.get("uploaded_at"), "uploaded_at"),
            published_at=_optional_datetime(payload.get("published_at"), "published_at"),
        ),
        version_detection_method=_required_string(payload, "version_detection_method"),
        matched_existing_document=bool(payload.get("matched_existing_document", False)),
    )


def stored_document_from_record(
    *,
    document: DocumentRecord,
    version: DocumentVersionRecord,
) -> StoredDocumentReference:
    metadata = dict(version.metadata or {})
    storage_uri = version.storage_uri or str(metadata.get("storage_uri") or "")
    if not storage_uri:
        raise ValueError("The selected document version has no durable storage URI.")
    original_filename = str(
        metadata.get("original_filename")
        or document.original_filename
        or Path(storage_uri).name
        or "document"
    )
    content_type = version.content_type or _optional_string(metadata.get("content_type")) or document.content_type
    checksum = version.checksum_sha256 or _optional_string(metadata.get("checksum_sha256")) or document.checksum_sha256
    if not checksum:
        raise ValueError("The selected document version has no checksum metadata.")
    size_bytes = metadata.get("size_bytes")
    if not isinstance(size_bytes, int):
        size_bytes = document.size_bytes
    if size_bytes is None:
        raise ValueError("The selected document version has no file-size metadata.")
    return StoredDocumentReference(
        storage_uri=storage_uri,
        original_filename=original_filename,
        content_type=content_type,
        size_bytes=int(size_bytes),
        checksum_sha256=checksum,
        storage_backend=str(metadata.get("storage_backend") or _infer_storage_backend(storage_uri)),
        bucket_name=_optional_string(metadata.get("bucket_name")),
        object_key=_optional_string(metadata.get("object_key")),
    )


def _stored_document_to_payload(reference: StoredDocumentReference) -> dict[str, Any]:
    return {
        "storage_uri": reference.storage_uri,
        "original_filename": reference.original_filename,
        "content_type": reference.content_type,
        "size_bytes": reference.size_bytes,
        "checksum_sha256": reference.checksum_sha256,
        "storage_backend": reference.storage_backend,
        "bucket_name": reference.bucket_name,
        "object_key": reference.object_key,
    }


def _stored_document_from_payload(payload: dict[str, Any]) -> StoredDocumentReference:
    return StoredDocumentReference(
        storage_uri=_required_string(payload, "storage_uri"),
        original_filename=_required_string(payload, "original_filename"),
        content_type=_optional_string(payload.get("content_type")),
        size_bytes=_positive_int(payload.get("size_bytes"), "size_bytes", allow_zero=True),
        checksum_sha256=_required_string(payload, "checksum_sha256"),
        storage_backend=_required_string(payload, "storage_backend"),
        bucket_name=_optional_string(payload.get("bucket_name")),
        object_key=_optional_string(payload.get("object_key")),
    )


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _optional_datetime(value: object, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string or null.")
    return datetime.fromisoformat(value)


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _positive_int(value: object, field: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    return value


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object.")
    return value


def _infer_storage_backend(storage_uri: str) -> str:
    return "minio" if storage_uri.startswith("minio://") else "local"
