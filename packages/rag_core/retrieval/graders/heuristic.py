from __future__ import annotations

import re
from collections.abc import Iterable

from packages.rag_core.retrieval.graders.models import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
)
from packages.rag_core.retrieval.models import EvidenceItem

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class HeuristicEvidenceGrader:
    """Deterministic lexical grader used when model-based grading is unavailable."""

    name = "heuristic_evidence_grader"

    def __init__(
        self,
        *,
        relevance_threshold: float = 0.35,
        sufficiency_threshold: float = 0.6,
        min_relevant_evidence: int = 1,
    ) -> None:
        _validate_probability(relevance_threshold, "relevance_threshold")
        _validate_probability(sufficiency_threshold, "sufficiency_threshold")
        if min_relevant_evidence <= 0:
            raise ValueError("min_relevant_evidence must be positive.")
        self._relevance_threshold = relevance_threshold
        self._sufficiency_threshold = sufficiency_threshold
        self._min_relevant_evidence = min_relevant_evidence

    async def grade(self, question: str, evidence: list[EvidenceItem]) -> EvidenceGradingReport:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")
        if not evidence:
            return EvidenceGradingReport(
                status=EvidenceSufficiency.MISSING,
                coverage_score=0.0,
                grades=(),
                rationale="No evidence was retrieved for the question.",
                grader_name=self.name,
            )

        question_terms = _content_terms(normalized)
        grades = tuple(self._grade_item(item, question_terms) for item in evidence)
        relevant_grades = tuple(grade for grade in grades if grade.relevant)
        coverage_score = max((grade.relevance_score for grade in grades), default=0.0)

        if not relevant_grades:
            status = EvidenceSufficiency.MISSING
            rationale = "None of the retrieved chunks overlap enough with the question to be considered relevant."
        elif (
            len(relevant_grades) >= self._min_relevant_evidence
            and coverage_score >= self._sufficiency_threshold
        ):
            status = EvidenceSufficiency.SUFFICIENT
            rationale = (
                f"{len(relevant_grades)} of {len(grades)} chunks are relevant and the strongest lexical "
                "coverage meets the sufficiency threshold."
            )
        else:
            status = EvidenceSufficiency.WEAK
            rationale = (
                f"{len(relevant_grades)} of {len(grades)} chunks are relevant, but their lexical coverage "
                "is too weak for a confident answer."
            )

        return EvidenceGradingReport(
            status=status,
            coverage_score=coverage_score,
            grades=grades,
            rationale=rationale,
            grader_name=self.name,
        )

    def _grade_item(self, item: EvidenceItem, question_terms: set[str]) -> EvidenceGrade:
        evidence_terms = _content_terms(item.text)
        if not question_terms or not evidence_terms:
            score = 0.0
        else:
            matched_terms = question_terms & evidence_terms
            query_coverage = len(matched_terms) / len(question_terms)
            score = min(1.0, query_coverage)

        relevant = score >= self._relevance_threshold
        rationale = (
            "The chunk contains enough question-specific terms."
            if relevant
            else "The chunk has limited lexical overlap with the question."
        )
        return EvidenceGrade(
            evidence_rank=item.rank,
            relevance_score=round(score, 4),
            relevant=relevant,
            rationale=rationale,
        )


def _content_terms(value: str) -> set[str]:
    return {token for token in _tokens(value) if len(token) > 1 and token not in _STOP_WORDS}


def _tokens(value: str) -> Iterable[str]:
    return (match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(value))


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
