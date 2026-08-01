from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.document_scope import DocumentScope
from packages.rag_core.documents import DocumentNameConstraint, DocumentVersionConstraint
from packages.rag_core.query_understanding.temporal import DocumentDateConstraint


@dataclass(frozen=True, slots=True)
class RetrievalConstraints:
    """Structured metadata constraints applied without changing semantic scores."""

    document: DocumentNameConstraint = DocumentNameConstraint()
    version: DocumentVersionConstraint = DocumentVersionConstraint()
    dates: tuple[DocumentDateConstraint, ...] = ()
    document_scope: DocumentScope = DocumentScope()

    @property
    def active(self) -> bool:
        return (
            self.document.active
            or self.version.active
            or bool(self.dates)
            or self.document_scope.strict
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "document": self.document.to_metadata(),
            "version": self.version.to_metadata(),
            "dates": [constraint.to_metadata() for constraint in self.dates],
            "document_scope": self.document_scope.to_metadata(),
            "active": self.active,
        }


@dataclass(slots=True)
class EvidenceItem:
    """A retrieved chunk snapshot used by query graphs and answer generation."""

    rank: int
    text: str
    score: float | None = None
    qdrant_chunk_index_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    subject_lane_id: str | None = None
    subject_id: uuid.UUID | None = None
    subject_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
