from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class UploadFile(Protocol):
    """Framework-neutral shape required from an uploaded file."""

    filename: str | None
    content_type: str | None

    async def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes from the upload stream."""


@dataclass(frozen=True, slots=True)
class StoredDocumentFile:
    """Metadata for a persisted source file and its parser-readable path."""

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
    async def save_upload(self, upload: UploadFile) -> StoredDocumentFile:
        """Persist an uploaded file and return parser-readable metadata."""

    def cleanup_staging_file(self, stored_file: StoredDocumentFile) -> None:
        """Remove temporary parser staging files if the backend uses them."""


class DocumentStorageError(RuntimeError):
    """Raised when a source document cannot be stored."""
