from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_SUBJECT_NAME_LENGTH = 255
_NON_ALNUM_PATTERN = re.compile(r"[\W_]+", re.UNICODE)


class InvalidSubjectNameError(ValueError):
    """A subject display name cannot produce a valid canonical lookup name."""


@dataclass(frozen=True, slots=True)
class SubjectName:
    """Validated display and lookup forms of a subject name or alias."""

    value: str
    normalized: str

    @classmethod
    def from_value(cls, value: str) -> SubjectName:
        display = _clean_display_name(value)
        if not display:
            raise InvalidSubjectNameError("subject name must not be empty.")
        if len(display) > MAX_SUBJECT_NAME_LENGTH:
            raise InvalidSubjectNameError(
                f"subject name must not exceed {MAX_SUBJECT_NAME_LENGTH} characters."
            )
        normalized = normalize_subject_name(display)
        if not normalized:
            raise InvalidSubjectNameError(
                "subject name must contain letters or numbers."
            )
        if len(normalized) > MAX_SUBJECT_NAME_LENGTH:
            raise InvalidSubjectNameError(
                "normalized subject name must not exceed "
                f"{MAX_SUBJECT_NAME_LENGTH} characters."
            )
        return cls(value=display, normalized=normalized)


def normalize_subject_name(value: str) -> str:
    """Return the single canonical lookup form used by domain and persistence."""

    cleaned = _clean_display_name(value)
    if not cleaned:
        return ""
    folded = unicodedata.normalize("NFKC", cleaned).casefold()
    normalized = _NON_ALNUM_PATTERN.sub(" ", folded)
    return " ".join(normalized.split())


def _clean_display_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())
