from __future__ import annotations

import re
from pathlib import Path

_SEPARATOR_PATTERN = re.compile(r"[\s._-]+")
_TRAILING_VERSION_MARKER = re.compile(
    r"(?:\s+(?:draft|version|ver|revision|rev|release|edition|copy)\s*\d+(?:\.\d+)*)$|"
    r"(?:\s+v\d+(?:\.\d+)*)$",
    re.IGNORECASE,
)


def normalized_document_identity(value: str) -> str:
    """Normalize a title or filename for exact logical-document matching."""

    stem = Path(value.strip()).stem.casefold()
    return _SEPARATOR_PATTERN.sub(" ", stem).strip()


def document_family_key(value: str) -> str:
    """Return a conservative family key with only trailing version markers removed.

    Examples: ``Realization_Draft4`` and ``Realization_Draft5`` both become
    ``realization``. Ordinary numbers that are not introduced by a known
    version marker are preserved.
    """

    normalized = normalized_document_identity(value)
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = _TRAILING_VERSION_MARKER.sub("", normalized).strip(" ._-")
    return _SEPARATOR_PATTERN.sub(" ", normalized).strip()


def same_document_family(left: str, right: str) -> bool:
    left_key = document_family_key(left)
    right_key = document_family_key(right)
    return bool(left_key and len(left_key) >= 4 and left_key == right_key)
