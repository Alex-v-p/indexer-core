from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EvidenceSufficiency(StrEnum):
    """Overall quality of the retrieved evidence for the submitted question."""

    MISSING = "missing"
    WEAK = "weak"
    SUFFICIENT = "sufficient"


@dataclass(frozen=True, slots=True)
class EvidenceGrade:
    """Question-specific relevance assessment for one retrieved evidence item."""

    evidence_rank: int
    relevance_score: float
    relevant: bool
    rationale: str

    def __post_init__(self) -> None:
        if self.evidence_rank <= 0:
            raise ValueError("evidence_rank must be positive.")
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("relevance_score must be between 0 and 1.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "evidence_rank": self.evidence_rank,
            "relevance_score": self.relevance_score,
            "relevant": self.relevant,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class EvidenceGradingReport:
    """Aggregate evidence-quality decision stored in QueryState and traces."""

    status: EvidenceSufficiency
    coverage_score: float
    grades: tuple[EvidenceGrade, ...]
    rationale: str
    grader_name: str
    fallback_used: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage_score <= 1.0:
            raise ValueError("coverage_score must be between 0 and 1.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.grader_name.strip():
            raise ValueError("grader_name must not be empty.")

        ranks = [grade.evidence_rank for grade in self.grades]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Evidence grades must contain unique evidence ranks.")
        if self.status is EvidenceSufficiency.MISSING and self.relevant_count:
            raise ValueError("Missing evidence cannot contain relevant grades.")
        if self.status is EvidenceSufficiency.SUFFICIENT and not self.relevant_count:
            raise ValueError("Sufficient evidence requires at least one relevant grade.")

    @property
    def sufficient(self) -> bool:
        return self.status is EvidenceSufficiency.SUFFICIENT

    @property
    def relevant_count(self) -> int:
        return sum(1 for grade in self.grades if grade.relevant)

    @property
    def total_count(self) -> int:
        return len(self.grades)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "coverage_score": self.coverage_score,
            "sufficient": self.sufficient,
            "missing_evidence": self.status is EvidenceSufficiency.MISSING,
            "weak_evidence": self.status is EvidenceSufficiency.WEAK,
            "relevant_count": self.relevant_count,
            "total_count": self.total_count,
            "rationale": self.rationale,
            "grader_name": self.grader_name,
            "fallback_used": self.fallback_used,
            "grades": [grade.to_metadata() for grade in self.grades],
        }
