from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem


@dataclass(slots=True)
class CitationItem:
    """A citation derived from retrieved evidence."""

    citation_index: int
    evidence_rank: int | None = None
    label: str | None = None
    page_number: int | None = None
    quote: str | None = None
    qdrant_chunk_index_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnswerGenerationRequest:
    """Inputs required to generate or safely block an answer."""

    question: str
    candidate_evidence: tuple[EvidenceItem, ...]
    evidence_grading: EvidenceGradingReport | None


@dataclass(frozen=True, slots=True)
class AnswerGenerationResult:
    """Generation result applied to QueryState by the orchestration node."""

    answer: str
    evidence: tuple[EvidenceItem, ...]
    citations: tuple[CitationItem, ...]
    candidate_evidence_count: int
    irrelevant_evidence_filtered_count: int
    unresolved_information: tuple[str, ...]
    supported_information: tuple[str, ...]
    is_partial: bool
    blocked_by_evidence_grading: bool
