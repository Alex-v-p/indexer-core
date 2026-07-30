from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from packages.rag_core.retrieval.constraint_validation import ConstraintValidationReport
from packages.rag_core.retrieval.evidence_context import EvidenceContextBundle
from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.models import RetrievalConstraints
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


class AnswerPresentationOutcome(StrEnum):
    """Stable renderer-owned outcome for presenting an answer."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    BLOCKED_CONSTRAINT_NO_MATCH = "blocked_constraint_no_match"
    BLOCKED_INSUFFICIENT_EVIDENCE = "blocked_insufficient_evidence"
    BLOCKED_NO_EVIDENCE = "blocked_no_evidence"


@dataclass(frozen=True, slots=True)
class AnswerPresentation:
    """Versioned structured presentation stored alongside the legacy answer."""

    outcome: AnswerPresentationOutcome
    title: str
    body: str
    supported_information: tuple[str, ...] = ()
    unresolved_information: tuple[str, ...] = ()
    citation_count: int = 0
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("Unsupported answer presentation schema version.")
        if not self.title.strip():
            raise ValueError("Answer presentation title must not be empty.")
        if self.citation_count < 0:
            raise ValueError("Answer presentation citation_count must not be negative.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome.value,
            "title": self.title,
            "body": self.body,
            "supported_information": list(self.supported_information),
            "unresolved_information": list(self.unresolved_information),
            "citation_count": self.citation_count,
        }


@dataclass(frozen=True, slots=True)
class AnswerGenerationRequest:
    """Inputs required to generate or safely block an answer."""

    question: str
    candidate_evidence: tuple[EvidenceItem, ...]
    evidence_grading: EvidenceGradingReport | None
    retrieval_constraints: RetrievalConstraints = RetrievalConstraints()
    constraint_validation: ConstraintValidationReport | None = None
    evidence_context: EvidenceContextBundle | None = None


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
    presentation: AnswerPresentation
