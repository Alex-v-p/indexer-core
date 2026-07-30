from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from packages.indexer_application.ports.object_storage import (
    DocumentStorageError,
    MaterializedDocumentFile,
    StoredDocumentReference,
    UploadFile,
)
from packages.indexer_infrastructure.object_storage.utils import raise_if_too_large, safe_filename

_DEFAULT_CONTENT_TYPE = "application/octet-stream"


class MinioDocumentObjectStore:
    """MinIO-backed durable source storage with explicit local materialization."""

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

    async def save_upload(self, upload: UploadFile) -> StoredDocumentReference:
        original_filename = Path(upload.filename or "document").name
        safe_name = safe_filename(original_filename)
        upload_id = uuid4().hex
        object_key = f"{self.object_prefix}/{upload_id}/{safe_name}" if self.object_prefix else f"{upload_id}/{safe_name}"
        staging_parent = self.staging_dir / f"upload-{upload_id}"
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
                    raise_if_too_large(size_bytes, self.max_size_bytes)
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
        except ValueError as exc:
            raise DocumentStorageError(str(exc)) from exc
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)

        return StoredDocumentReference(
            storage_uri=f"s3://{self.bucket_name}/{object_key}",
            original_filename=original_filename,
            content_type=upload.content_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum.hexdigest(),
            storage_backend="minio",
            bucket_name=self.bucket_name,
            object_key=object_key,
        )

    async def materialize(self, reference: StoredDocumentReference) -> MaterializedDocumentFile:
        if reference.storage_backend != "minio" or not reference.bucket_name or not reference.object_key:
            raise DocumentStorageError("MinIO storage requires a MinIO document reference.")
        materialization_id = uuid4().hex
        parent = self.staging_dir / f"materialized-{materialization_id}"
        parent.mkdir(parents=True, exist_ok=False)
        path = parent / safe_filename(reference.original_filename)
        try:
            self.client.fget_object(reference.bucket_name, reference.object_key, str(path))
        except Exception:
            shutil.rmtree(parent, ignore_errors=True)
            raise
        return MaterializedDocumentFile(reference=reference, path=path)

    def cleanup_materialized_file(self, materialized: MaterializedDocumentFile) -> None:
        if materialized.storage_backend == "minio":
            shutil.rmtree(materialized.path.parent, ignore_errors=True)

    def _ensure_bucket_exists(self) -> None:
        if self.client.bucket_exists(self.bucket_name):
            return
        self.client.make_bucket(self.bucket_name, location=self.region)
