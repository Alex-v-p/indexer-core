from __future__ import annotations

import re
from pathlib import Path

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "document"
    cleaned = _SAFE_FILENAME_RE.sub("_", name)
    return cleaned[:180] or "document"


def raise_if_too_large(size_bytes: int, max_size_bytes: int | None) -> None:
    if max_size_bytes is None or size_bytes <= max_size_bytes:
        return

    size_mb = max_size_bytes / 1024 / 1024
    raise ValueError(f"Uploaded file exceeds the {size_mb:g} MB limit.")
