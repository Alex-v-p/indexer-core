from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


@dataclass(frozen=True, slots=True)
class StoredDocumentFile:
    """Metadata for a source file persisted by the document store."""

    path: Path
    storage_uri: str
    original_filename: str
    content_type: str | None
    size_bytes: int
    checksum_sha256: str
    storage_backend: str
    bucket_name: str | None = None
    object_key: str | None = None


class DocumentStorage(Protocol):
    """Storage boundary for uploaded source documents."""

    async def save_upload(self, upload: UploadFile) -> StoredDocumentFile:
        """Persist an uploaded file and return parser-readable staging metadata."""

    def cleanup_staging_file(self, stored_file: StoredDocumentFile) -> None:
        """Remove temporary parser staging files if the backend uses them."""


class DocumentStorageError(RuntimeError):
    """Raised when a source document cannot be stored."""


class LocalDocumentStorage:
    """Local filesystem storage for test/offline development."""

    def __init__(self, *, base_dir: str | Path, max_size_bytes: int | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.max_size_bytes = max_size_bytes

    async def save_upload(self, upload: UploadFile) -> StoredDocumentFile:
        original_filename = Path(upload.filename or "document").name
        safe_name = _safe_filename(original_filename)
        document_dir = self.base_dir / uuid4().hex
        document_dir.mkdir(parents=True, exist_ok=False)
        destination = document_dir / safe_name

        checksum = hashlib.sha256()
        size_bytes = 0
        try:
            with destination.open("wb") as output:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    _raise_if_too_large(size_bytes, self.max_size_bytes)
                    checksum.update(chunk)
                    output.write(chunk)
        except Exception:
            shutil.rmtree(document_dir, ignore_errors=True)
            raise

        return StoredDocumentFile(
            path=destination,
            storage_uri=f"file://{destination}",
            original_filename=original_filename,
            content_type=upload.content_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum.hexdigest(),
            storage_backend="local",
        )

    def cleanup_staging_file(self, stored_file: StoredDocumentFile) -> None:
        return None


class MinioDocumentStorage:
    """MinIO-backed storage for durable uploaded source documents.

    The uploaded object is staged briefly on disk because the parser boundary is
    intentionally file-path based in Phase 1. The durable copy is the MinIO
    object; the staging file is deleted by the ingestion service after parsing.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False,
        region: str | None = None,
        object_prefix: str = "documents",
        staging_dir: str | Path = "/tmp/indexer-ingestion",
        max_size_bytes: int | None = None,
    ) -> None:
        try:
            from minio import Minio
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DocumentStorageError("Install the 'minio' package to use MinIO document storage.") from exc

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
        )
        self.bucket_name = bucket_name
        self.region = region
        self.object_prefix = object_prefix.strip("/")
        self.staging_dir = Path(staging_dir)
        self.max_size_bytes = max_size_bytes

    async def save_upload(self, upload: UploadFile) -> StoredDocumentFile:
        original_filename = Path(upload.filename or "document").name
        safe_name = _safe_filename(original_filename)
        upload_id = uuid4().hex
        object_key = f"{self.object_prefix}/{upload_id}/{safe_name}" if self.object_prefix else f"{upload_id}/{safe_name}"
        staging_parent = self.staging_dir / upload_id
        staging_parent.mkdir(parents=True, exist_ok=False)
        staging_path = staging_parent / safe_name

        checksum = hashlib.sha256()
        size_bytes = 0
        try:
            with staging_path.open("wb") as output:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    _raise_if_too_large(size_bytes, self.max_size_bytes)
                    checksum.update(chunk)
                    output.write(chunk)

            self._ensure_bucket_exists()
            self.client.fput_object(
                self.bucket_name,
                object_key,
                str(staging_path),
                content_type=upload.content_type or _DEFAULT_CONTENT_TYPE,
                metadata={"sha256": checksum.hexdigest()},
            )
        except Exception:
            shutil.rmtree(staging_parent, ignore_errors=True)
            raise

        return StoredDocumentFile(
            path=staging_path,
            storage_uri=f"s3://{self.bucket_name}/{object_key}",
            original_filename=original_filename,
            content_type=upload.content_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum.hexdigest(),
            storage_backend="minio",
            bucket_name=self.bucket_name,
            object_key=object_key,
        )

    def cleanup_staging_file(self, stored_file: StoredDocumentFile) -> None:
        if stored_file.storage_backend != "minio":
            return
        shutil.rmtree(stored_file.path.parent, ignore_errors=True)

    def _ensure_bucket_exists(self) -> None:
        if self.client.bucket_exists(self.bucket_name):
            return
        self.client.make_bucket(self.bucket_name, location=self.region)


def build_document_storage(settings: Settings) -> DocumentStorage:
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024 if settings.max_upload_size_mb > 0 else None
    if settings.document_storage_backend == "local":
        return LocalDocumentStorage(base_dir=settings.document_storage_dir, max_size_bytes=max_size_bytes)

    return MinioDocumentStorage(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name=settings.minio_bucket_name,
        secure=settings.minio_secure,
        region=settings.minio_region or None,
        object_prefix=settings.minio_object_prefix,
        staging_dir=settings.document_staging_dir,
        max_size_bytes=max_size_bytes,
    )


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "document"
    cleaned = _SAFE_FILENAME_RE.sub("_", name)
    return cleaned[:180] or "document"


def _raise_if_too_large(size_bytes: int, max_size_bytes: int | None) -> None:
    if max_size_bytes is not None and size_bytes > max_size_bytes:
        size_mb = max_size_bytes / 1024 / 1024
        raise DocumentStorageError(f"Uploaded file exceeds the {size_mb:g} MB limit.")
