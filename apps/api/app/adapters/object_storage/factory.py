from __future__ import annotations

from app.adapters.object_storage.base import DocumentObjectStore
from app.adapters.object_storage.local import LocalDocumentObjectStore
from app.adapters.object_storage.minio import MinioDocumentObjectStore
from app.core.config import Settings


def build_document_object_store(settings: Settings) -> DocumentObjectStore:
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024 if settings.max_upload_size_mb > 0 else None
    if settings.document_storage_backend == "local":
        return LocalDocumentObjectStore(base_dir=settings.document_storage_dir, max_size_bytes=max_size_bytes)

    return MinioDocumentObjectStore(
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
