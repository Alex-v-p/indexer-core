from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Mapping

_KNOWN_EXTENSION_PATTERN = re.compile(
    r"\.(?:pdf|md|markdown|txt|doc|docx|rtf|odt|html?|csv|json|ya?ml)$",
    re.IGNORECASE,
)
_NON_ALNUM_PATTERN = re.compile(r"[\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class DocumentNameConstraint:
    """Exact logical-document names explicitly requested by the user.

    Names are matched case-insensitively after conservative punctuation and
    file-extension normalization. Multiple names use OR semantics so a query
    can explicitly request more than one document.
    """

    names: tuple[str, ...] = ()
    confidence: float = 0.0
    rationale: str = "No document-name constraint was detected."
    detector_name: str = "none"

    def __post_init__(self) -> None:
        cleaned = tuple(_clean_display_name(name) for name in self.names)
        if any(not name for name in cleaned):
            raise ValueError("document names must not be empty.")
        normalized = tuple(normalize_document_name(name) for name in cleaned)
        if any(not name for name in normalized):
            raise ValueError("document names must contain searchable characters.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("document names must be unique after normalization.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.detector_name.strip():
            raise ValueError("detector_name must not be empty.")
        object.__setattr__(self, "names", cleaned)

    @property
    def active(self) -> bool:
        return bool(self.names)

    @property
    def normalized_names(self) -> tuple[str, ...]:
        return tuple(normalize_document_name(name) for name in self.names)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "normalized_names": list(self.normalized_names),
            "active": self.active,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "detector_name": self.detector_name,
            "match_semantics": "exact_normalized_any",
        }


def normalize_document_name(value: str) -> str:
    """Normalize a title or filename without broad/fuzzy matching."""

    cleaned = _clean_display_name(value)
    if not cleaned:
        return ""
    basename = PurePath(cleaned.replace("\\", "/")).name
    basename = _KNOWN_EXTENSION_PATTERN.sub("", basename)
    normalized = _NON_ALNUM_PATTERN.sub(" ", basename.casefold())
    return " ".join(normalized.split())


def evidence_document_name_matches(
    metadata: Mapping[str, object],
    constraint: DocumentNameConstraint,
) -> bool:
    if not constraint.active:
        return True

    allowed = set(constraint.normalized_names)
    available = document_name_metadata_values(metadata)
    return bool(allowed.intersection(available))


def document_name_metadata_values(metadata: Mapping[str, object]) -> set[str]:
    values: set[str] = set()
    for key in (
        "document_title_normalized",
        "original_filename_normalized",
        "document_title",
        "original_filename",
        "filename",
        "source_name",
    ):
        raw_value = metadata.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        normalized = (
            " ".join(raw_value.casefold().split())
            if key.endswith("_normalized")
            else normalize_document_name(raw_value)
        )
        if normalized:
            values.add(normalized)
    return values


def _clean_display_name(value: str) -> str:
    cleaned = " ".join(str(value).strip().split())
    pairs = (("\"", "\""), ("'", "'"), ("`", "`"), ("“", "”"), ("‘", "’"))
    for start, end in pairs:
        if len(cleaned) >= 2 and cleaned.startswith(start) and cleaned.endswith(end):
            cleaned = cleaned[len(start) : len(cleaned) - len(end)].strip()
            break
    return cleaned
