from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import UploadFile


@dataclass(frozen=True, slots=True)
class StoredDocumentFile:
    """Metadata for a source file persisted by a document object store.

    ``path`` is a parser-readable local staging path. It may point to the
    durable file itself for local storage, or to a temporary copy for MinIO.
    """

    path: Path
    storage_uri: str
    original_filename: str
    content_type: str | None
    size_bytes: int
    checksum_sha256: str
    storage_backend: str
    bucket_name: str | None = None
    object_key: str | None = None


class DocumentObjectStore(Protocol):
    """Storage boundary for uploaded source documents."""

    async def save_upload(self, upload: UploadFile) -> StoredDocumentFile:
        """Persist an uploaded file and return parser-readable staging metadata."""

    def cleanup_staging_file(self, stored_file: StoredDocumentFile) -> None:
        """Remove temporary parser staging files if the backend uses them."""


class DocumentStorageError(RuntimeError):
    """Raised when a source document cannot be stored."""
