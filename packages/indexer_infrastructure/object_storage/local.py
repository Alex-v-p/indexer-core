from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from packages.indexer_application.ports.object_storage import DocumentStorageError, StoredDocumentFile, UploadFile
from packages.indexer_infrastructure.object_storage.utils import raise_if_too_large, safe_filename


class LocalDocumentObjectStore:
    """Local filesystem storage for tests and offline development."""

    def __init__(self, *, base_dir: str | Path, max_size_bytes: int | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.max_size_bytes = max_size_bytes

    async def save_upload(self, upload: UploadFile) -> StoredDocumentFile:
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
