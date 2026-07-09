from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


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
