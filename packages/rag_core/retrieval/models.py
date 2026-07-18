from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.documents import DocumentVersionConstraint


@dataclass(frozen=True, slots=True)
class RetrievalConstraints:
    """Structured metadata constraints applied without changing semantic scores."""

    version: DocumentVersionConstraint = DocumentVersionConstraint()

    @property
    def active(self) -> bool:
        return self.version.active

    def to_metadata(self) -> dict[str, Any]:
        return {"version": self.version.to_metadata(), "active": self.active}


@dataclass(slots=True)
class EvidenceItem:
    """A retrieved chunk snapshot used by query graphs and answer generation."""

    rank: int
    text: str
    score: float | None = None
    qdrant_chunk_index_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
