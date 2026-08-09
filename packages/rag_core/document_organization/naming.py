from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_CONTENT_GROUP_NAME_LENGTH = 80
MAX_DOCUMENT_TYPE_KEY_LENGTH = 64
_NON_ALNUM_PATTERN = re.compile(r"[\W_]+", re.UNICODE)
_TYPE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_AUTOMATIC_NAME_CONNECTORS = {
    "a", "an", "and", "at", "for", "from", "in", "of", "on", "the", "to", "with",
}


class InvalidContentGroupNameError(ValueError):
    """A content-group display name cannot produce a stable catalogue name."""


class InvalidDocumentTypeKeyError(ValueError):
    """A document-type key is not a stable lower-snake-case identifier."""


@dataclass(frozen=True, slots=True)
class ContentGroupName:
    value: str
    normalized: str

    @classmethod
    def from_value(cls, value: str) -> ContentGroupName:
        display = _clean(value)
        if not display:
            raise InvalidContentGroupNameError("content group name must not be empty.")
        if len(display) > MAX_CONTENT_GROUP_NAME_LENGTH:
            raise InvalidContentGroupNameError(
                f"content group name must not exceed {MAX_CONTENT_GROUP_NAME_LENGTH} characters."
            )
        normalized = normalize_content_group_name(display)
        if not normalized:
            raise InvalidContentGroupNameError(
                "content group name must contain letters or numbers."
            )
        return cls(display, normalized)

    @classmethod
    def from_automatic_proposal(cls, value: str) -> ContentGroupName:
        name = cls.from_value(value)
        meaningful = tuple(
            token
            for token in name.normalized.split()
            if token not in _AUTOMATIC_NAME_CONNECTORS
        )
        if not 2 <= len(meaningful) <= 6:
            raise InvalidContentGroupNameError(
                "automatic content group names require 2 to 6 meaningful words."
            )
        return name


@dataclass(frozen=True, slots=True)
class DocumentTypeKey:
    value: str

    @classmethod
    def from_value(cls, value: str) -> DocumentTypeKey:
        cleaned = str(value).strip()
        if len(cleaned) > MAX_DOCUMENT_TYPE_KEY_LENGTH or not _TYPE_KEY_PATTERN.fullmatch(cleaned):
            raise InvalidDocumentTypeKeyError(
                "document type key must be lower-snake-case and no longer than 64 characters."
            )
        return cls(cleaned)


def normalize_content_group_name(value: str) -> str:
    cleaned = _clean(value)
    if not cleaned:
        return ""
    folded = unicodedata.normalize("NFKC", cleaned).casefold()
    return " ".join(_NON_ALNUM_PATTERN.sub(" ", folded).split())


def _clean(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())
