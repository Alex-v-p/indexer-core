from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from packages.indexer_application.ports.object_storage import (
    DocumentStorageError,
    MaterializedDocumentFile,
    StoredDocumentReference,
    UploadFile,
)
from packages.indexer_infrastructure.object_storage.utils import raise_if_too_large, safe_filename


class LocalDocumentObjectStore:
    """Local filesystem storage for tests and offline development."""

    def __init__(self, *, base_dir: str | Path, max_size_bytes: int | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.max_size_bytes = max_size_bytes

    async def save_upload(self, upload: UploadFile) -> StoredDocumentReference:
        original_filename = Path(upload.filename or "document").name
        safe_name = safe_filename(original_filename)
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
                    raise_if_too_large(size_bytes, self.max_size_bytes)
                    checksum.update(chunk)
                    output.write(chunk)
        except ValueError as exc:
            shutil.rmtree(document_dir, ignore_errors=True)
            raise DocumentStorageError(str(exc)) from exc
        except Exception:
            shutil.rmtree(document_dir, ignore_errors=True)
            raise

        return StoredDocumentReference(
            storage_uri=f"file://{destination}",
            original_filename=original_filename,
            content_type=upload.content_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum.hexdigest(),
            storage_backend="local",
            object_key=str(destination),
        )

    async def materialize(self, reference: StoredDocumentReference) -> MaterializedDocumentFile:
        if reference.storage_backend != "local":
            raise DocumentStorageError("Local storage cannot materialize a non-local document reference.")
        if reference.object_key:
            path = Path(reference.object_key)
        else:
            parsed = urlparse(reference.storage_uri)
            path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(reference.storage_uri)
        if not path.is_file():
            raise DocumentStorageError(f"Stored document is not available at {reference.storage_uri}.")
        return MaterializedDocumentFile(reference=reference, path=path)

    async def delete(self, reference: StoredDocumentReference) -> None:
        if reference.storage_backend != "local":
            raise DocumentStorageError("Local storage cannot delete a non-local document reference.")
        if reference.object_key:
            path = Path(reference.object_key)
        else:
            parsed = urlparse(reference.storage_uri)
            path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(reference.storage_uri)
        try:
            path.unlink(missing_ok=True)
            if path.parent != self.base_dir and path.parent.exists():
                path.parent.rmdir()
        except OSError as exc:
            raise DocumentStorageError(
                f"Could not delete stored document {reference.storage_uri}: {exc}"
            ) from exc

    def cleanup_materialized_file(self, materialized: MaterializedDocumentFile) -> None:
        del materialized
