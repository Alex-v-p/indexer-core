from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EvidenceSufficiency(StrEnum):
    """Overall quality of the retrieved evidence for all required information needs."""

    MISSING = "missing"
    WEAK = "weak"
    SUFFICIENT = "sufficient"


class InformationNeedSupport(StrEnum):
    """How completely retrieved evidence supports one planned answer requirement."""

    MISSING = "missing"
    PARTIAL = "partial"
    SUPPORTED = "supported"


@dataclass(frozen=True, slots=True)
class EvidenceGrade:
    """Question-specific relevance assessment for one retrieved evidence item."""

    evidence_rank: int
    relevance_score: float
    relevant: bool
    rationale: str
    supports_information_need_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.evidence_rank <= 0:
            raise ValueError("evidence_rank must be positive.")
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("relevance_score must be between 0 and 1.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if len(self.supports_information_need_ids) != len(set(self.supports_information_need_ids)):
            raise ValueError("supports_information_need_ids must be unique.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "evidence_rank": self.evidence_rank,
            "relevance_score": self.relevance_score,
            "relevant": self.relevant,
            "rationale": self.rationale,
            "supports_information_need_ids": list(self.supports_information_need_ids),
        }


@dataclass(frozen=True, slots=True)
class InformationNeedGrade:
    """Coverage decision for one atomic requirement from the retrieval plan."""

    information_need_id: str
    description: str
    status: InformationNeedSupport
    coverage_score: float
    supporting_evidence_ranks: tuple[int, ...]
    rationale: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.information_need_id.strip():
            raise ValueError("information_need_id must not be empty.")
        if not self.description.strip():
            raise ValueError("description must not be empty.")
        if not 0.0 <= self.coverage_score <= 1.0:
            raise ValueError("coverage_score must be between 0 and 1.")
        if any(rank <= 0 for rank in self.supporting_evidence_ranks):
            raise ValueError("supporting_evidence_ranks must be positive.")
        if len(self.supporting_evidence_ranks) != len(set(self.supporting_evidence_ranks)):
            raise ValueError("supporting_evidence_ranks must be unique.")
        if self.status is InformationNeedSupport.MISSING and self.supporting_evidence_ranks:
            raise ValueError("Missing information needs cannot have supporting evidence ranks.")
        if self.status is not InformationNeedSupport.MISSING and not self.supporting_evidence_ranks:
            raise ValueError("Partial or supported information needs require supporting evidence ranks.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")

    @property
    def supported(self) -> bool:
        return self.status is InformationNeedSupport.SUPPORTED

    def to_metadata(self) -> dict[str, Any]:
        return {
            "information_need_id": self.information_need_id,
            "description": self.description,
            "status": self.status.value,
            "coverage_score": self.coverage_score,
            "supporting_evidence_ranks": list(self.supporting_evidence_ranks),
            "rationale": self.rationale,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class EvidenceGradingReport:
    """Chunk relevance plus claim/aspect-level coverage stored in state and traces."""

    status: EvidenceSufficiency
    coverage_score: float
    grades: tuple[EvidenceGrade, ...]
    rationale: str
    grader_name: str
    information_need_grades: tuple[InformationNeedGrade, ...] = ()
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
        need_ids = [grade.information_need_id for grade in self.information_need_grades]
        if len(need_ids) != len(set(need_ids)):
            raise ValueError("Information-need grades must contain unique ids.")
        known_ranks = set(ranks)
        if any(
            rank not in known_ranks
            for grade in self.information_need_grades
            for rank in grade.supporting_evidence_ranks
        ):
            raise ValueError("Information-need grades reference unknown evidence ranks.")
        if self.status is EvidenceSufficiency.MISSING and self.relevant_count:
            raise ValueError("Missing evidence cannot contain relevant grades.")
        if self.status is EvidenceSufficiency.SUFFICIENT and not self.relevant_count:
            raise ValueError("Sufficient evidence requires at least one relevant grade.")
        required_need_grades = tuple(grade for grade in self.information_need_grades if grade.required)
        if self.status is EvidenceSufficiency.SUFFICIENT and required_need_grades:
            if any(not grade.supported for grade in required_need_grades):
                raise ValueError("Sufficient evidence requires every required information need to be supported.")

    @property
    def sufficient(self) -> bool:
        return self.status is EvidenceSufficiency.SUFFICIENT

    @property
    def relevant_count(self) -> int:
        return sum(1 for grade in self.grades if grade.relevant)

    @property
    def total_count(self) -> int:
        return len(self.grades)

    @property
    def relevant_evidence_ranks(self) -> tuple[int, ...]:
        """Ranks approved by the grader for downstream answer generation."""

        return tuple(grade.evidence_rank for grade in self.grades if grade.relevant)

    @property
    def supported_information_need_count(self) -> int:
        return sum(1 for grade in self.information_need_grades if grade.supported)

    @property
    def supported_required_information_need_count(self) -> int:
        return sum(1 for grade in self.information_need_grades if grade.required and grade.supported)

    @property
    def required_information_need_count(self) -> int:
        return sum(1 for grade in self.information_need_grades if grade.required)

    @property
    def partial_answer_available(self) -> bool:
        """Whether at least one required claim can be answered despite incomplete coverage."""

        return not self.sufficient and self.supported_required_information_need_count > 0

    @property
    def answerable(self) -> bool:
        """Whether generation may produce a complete or explicitly partial answer."""

        return self.sufficient or self.partial_answer_available

    @property
    def supported_information(self) -> tuple[str, ...]:
        return tuple(
            grade.description
            for grade in self.information_need_grades
            if grade.required and grade.supported
        )

    @property
    def partial_information_need_count(self) -> int:
        return sum(1 for grade in self.information_need_grades if grade.status is InformationNeedSupport.PARTIAL)

    @property
    def missing_information_need_count(self) -> int:
        return sum(1 for grade in self.information_need_grades if grade.status is InformationNeedSupport.MISSING)

    @property
    def unresolved_information(self) -> tuple[str, ...]:
        return tuple(
            grade.description
            for grade in self.information_need_grades
            if grade.required and not grade.supported
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "coverage_score": self.coverage_score,
            "sufficient": self.sufficient,
            "answerable": self.answerable,
            "partial_answer_available": self.partial_answer_available,
            "missing_evidence": self.status is EvidenceSufficiency.MISSING,
            "weak_evidence": self.status is EvidenceSufficiency.WEAK,
            "relevant_count": self.relevant_count,
            "total_count": self.total_count,
            "relevant_evidence_ranks": list(self.relevant_evidence_ranks),
            "supported_information_need_count": self.supported_information_need_count,
            "supported_required_information_need_count": self.supported_required_information_need_count,
            "required_information_need_count": self.required_information_need_count,
            "supported_information": list(self.supported_information),
            "partial_information_need_count": self.partial_information_need_count,
            "missing_information_need_count": self.missing_information_need_count,
            "total_information_need_count": len(self.information_need_grades),
            "unresolved_information": list(self.unresolved_information),
            "rationale": self.rationale,
            "grader_name": self.grader_name,
            "fallback_used": self.fallback_used,
            "grades": [grade.to_metadata() for grade in self.grades],
            "information_need_grades": [grade.to_metadata() for grade in self.information_need_grades],
        }
