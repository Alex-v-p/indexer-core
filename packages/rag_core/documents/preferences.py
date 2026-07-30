from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from packages.rag_core.documents.naming import document_name_metadata_values


@dataclass(frozen=True, slots=True)
class DocumentReference:
    """Stable logical-document identity extracted from one evidence payload."""

    key: str
    display_name: str
    document_id: uuid.UUID | None = None
    document_version_ids: tuple[uuid.UUID, ...] = ()
    normalized_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("key must not be empty.")
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty.")
        if len(self.document_version_ids) != len(set(self.document_version_ids)):
            raise ValueError("document_version_ids must be unique.")
        normalized = tuple(dict.fromkeys(name for name in self.normalized_names if name.strip()))
        object.__setattr__(self, "normalized_names", normalized)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "document_id": str(self.document_id) if self.document_id is not None else None,
            "document_version_ids": [str(value) for value in self.document_version_ids],
            "normalized_names": list(self.normalized_names),
        }


@dataclass(frozen=True, slots=True)
class DocumentPreference:
    """A soft primary-document signal learned from already graded evidence."""

    document: DocumentReference
    score: float
    confidence: float
    margin: float
    supporting_information_need_ids: tuple[str, ...]
    supporting_evidence_ranks: tuple[int, ...]
    rationale: str
    detector_name: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not 0.0 <= self.margin <= 1.0:
            raise ValueError("margin must be between 0 and 1.")
        if any(not value.strip() for value in self.supporting_information_need_ids):
            raise ValueError("supporting_information_need_ids must not contain empty values.")
        if len(self.supporting_information_need_ids) != len(set(self.supporting_information_need_ids)):
            raise ValueError("supporting_information_need_ids must be unique.")
        if any(rank <= 0 for rank in self.supporting_evidence_ranks):
            raise ValueError("supporting_evidence_ranks must be positive.")
        if len(self.supporting_evidence_ranks) != len(set(self.supporting_evidence_ranks)):
            raise ValueError("supporting_evidence_ranks must be unique.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.detector_name.strip():
            raise ValueError("detector_name must not be empty.")

    def matches(
        self,
        *,
        document_id: uuid.UUID | None,
        document_version_id: uuid.UUID | None,
        metadata: Mapping[str, object],
    ) -> bool:
        if self.document.document_id is not None and document_id is not None:
            return self.document.document_id == document_id
        if document_version_id is not None and document_version_id in self.document.document_version_ids:
            return True
        return bool(set(self.document.normalized_names).intersection(document_name_metadata_values(metadata)))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "document": self.document.to_metadata(),
            "score": self.score,
            "confidence": self.confidence,
            "margin": self.margin,
            "supporting_information_need_ids": list(self.supporting_information_need_ids),
            "supporting_evidence_ranks": list(self.supporting_evidence_ranks),
            "rationale": self.rationale,
            "detector_name": self.detector_name,
            "semantics": "soft_preference_not_filter",
        }


def document_reference_from_values(
    *,
    rank: int,
    document_id: uuid.UUID | None,
    document_version_id: uuid.UUID | None,
    metadata: Mapping[str, object],
) -> DocumentReference:
    normalized_names = tuple(sorted(document_name_metadata_values(metadata)))
    display_name = _display_document_name(metadata, normalized_names, rank)
    if document_id is not None:
        key = f"document:{document_id}"
    elif normalized_names:
        key = f"name:{normalized_names[0]}"
    elif document_version_id is not None:
        key = f"version:{document_version_id}"
    else:
        key = f"unknown:{rank}"
    return DocumentReference(
        key=key,
        display_name=display_name,
        document_id=document_id,
        document_version_ids=(document_version_id,) if document_version_id is not None else (),
        normalized_names=normalized_names,
    )


def merge_document_references(left: DocumentReference, right: DocumentReference) -> DocumentReference:
    if left.key != right.key:
        raise ValueError("Only references for the same document key can be merged.")
    return DocumentReference(
        key=left.key,
        display_name=(
            left.display_name
            if not left.display_name.startswith("Unknown document")
            else right.display_name
        ),
        document_id=left.document_id or right.document_id,
        document_version_ids=tuple(dict.fromkeys((*left.document_version_ids, *right.document_version_ids))),
        normalized_names=tuple(dict.fromkeys((*left.normalized_names, *right.normalized_names))),
    )


def _display_document_name(
    metadata: Mapping[str, object],
    normalized_names: tuple[str, ...],
    rank: int,
) -> str:
    for key in ("document_title", "original_filename", "filename", "source_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.strip().split())
    if normalized_names:
        return normalized_names[0]
    return f"Unknown document for evidence {rank}"
