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
class StoredDocumentReference:
    """Stable identity and metadata for a durably stored source document."""

    storage_uri: str
    original_filename: str
    content_type: str | None
    size_bytes: int
    checksum_sha256: str
    storage_backend: str
    bucket_name: str | None = None
    object_key: str | None = None


@dataclass(frozen=True, slots=True)
class MaterializedDocumentFile:
    """Temporary parser-readable materialization of a stored document."""

    reference: StoredDocumentReference
    path: Path

    @property
    def storage_uri(self) -> str:
        return self.reference.storage_uri

    @property
    def original_filename(self) -> str:
        return self.reference.original_filename

    @property
    def content_type(self) -> str | None:
        return self.reference.content_type

    @property
    def size_bytes(self) -> int:
        return self.reference.size_bytes

    @property
    def checksum_sha256(self) -> str:
        return self.reference.checksum_sha256

    @property
    def storage_backend(self) -> str:
        return self.reference.storage_backend

    @property
    def bucket_name(self) -> str | None:
        return self.reference.bucket_name

    @property
    def object_key(self) -> str | None:
        return self.reference.object_key


@dataclass(frozen=True, slots=True)
class StoredDocumentFile:
    """Compatibility shape retained for older tests and helper callers."""

    path: Path
    storage_uri: str
    original_filename: str
    content_type: str | None
    size_bytes: int
    checksum_sha256: str
    storage_backend: str
    bucket_name: str | None = None
    object_key: str | None = None

    @property
    def reference(self) -> StoredDocumentReference:
        return StoredDocumentReference(
            storage_uri=self.storage_uri,
            original_filename=self.original_filename,
            content_type=self.content_type,
            size_bytes=self.size_bytes,
            checksum_sha256=self.checksum_sha256,
            storage_backend=self.storage_backend,
            bucket_name=self.bucket_name,
            object_key=self.object_key,
        )


class DocumentObjectStore(Protocol):
    async def save_upload(self, upload: UploadFile) -> StoredDocumentReference:
        """Persist an upload and return a stable durable reference."""

    async def materialize(self, reference: StoredDocumentReference) -> MaterializedDocumentFile:
        """Return a parser-readable local materialization of ``reference``."""

    async def delete(self, reference: StoredDocumentReference) -> None:
        """Delete a durably stored source document if it still exists."""

    def cleanup_materialized_file(self, materialized: MaterializedDocumentFile) -> None:
        """Release temporary parser materialization owned by the caller."""


class DocumentStorageError(RuntimeError):
    """Raised when a source document cannot be stored or materialized."""
