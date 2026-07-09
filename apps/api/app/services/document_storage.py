from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class StoredDocumentFile:
    """Metadata for a file persisted by the local document store."""

    path: Path
    storage_uri: str
    original_filename: str
    content_type: str | None
    size_bytes: int
    checksum_sha256: str


class LocalDocumentStorage:
    """Local filesystem storage for uploaded source documents."""

    def __init__(self, *, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    async def save_upload(self, upload: UploadFile) -> StoredDocumentFile:
        original_filename = Path(upload.filename or "document").name
        safe_name = _safe_filename(original_filename)
        document_dir = self.base_dir / uuid4().hex
        document_dir.mkdir(parents=True, exist_ok=False)
        destination = document_dir / safe_name

        checksum = hashlib.sha256()
        size_bytes = 0
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                checksum.update(chunk)
                output.write(chunk)

        return StoredDocumentFile(
            path=destination,
            storage_uri=f"file://{destination}",
            original_filename=original_filename,
            content_type=upload.content_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum.hexdigest(),
        )


def build_document_storage(settings: Settings) -> LocalDocumentStorage:
    return LocalDocumentStorage(base_dir=settings.document_storage_dir)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "document"
    cleaned = _SAFE_FILENAME_RE.sub("_", name)
    return cleaned[:180] or "document"
