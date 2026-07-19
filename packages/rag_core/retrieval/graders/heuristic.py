from __future__ import annotations

import re
from collections.abc import Iterable

from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.graders.models import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with",
}


class HeuristicEvidenceGrader:
    """Deterministic lexical fallback with per-information-need coverage."""

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

    async def grade(
        self,
        question: str,
        evidence: list[EvidenceItem],
        *,
        constraints: RetrievalConstraints | None = None,
    ) -> EvidenceGradingReport:
        return await self.grade_information_needs(
            question,
            evidence,
            (_single_information_need(question),),
            constraints=constraints,
        )

    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
        *,
        constraints: RetrievalConstraints | None = None,
    ) -> EvidenceGradingReport:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")
        needs = information_needs or (_single_information_need(normalized),)
        if not evidence:
            need_grades = tuple(
                InformationNeedGrade(
                    information_need_id=need.need_id,
                    description=need.description,
                    status=InformationNeedSupport.MISSING,
                    coverage_score=0.0,
                    supporting_evidence_ranks=(),
                    rationale="No evidence was retrieved for this information need.",
                    required=need.required,
                )
                for need in needs
            )
            return EvidenceGradingReport(
                status=EvidenceSufficiency.MISSING,
                coverage_score=0.0,
                grades=(),
                information_need_grades=need_grades,
                rationale="No evidence was retrieved for the question.",
                grader_name=self.name,
            )

        per_need_scores = {
            need.need_id: {
                item.rank: _lexical_coverage(need.retrieval_query or need.description, item.text)
                for item in evidence
            }
            for need in needs
        }
        need_grades = tuple(self._grade_need(need, per_need_scores[need.need_id]) for need in needs)
        evidence_grades = tuple(
            self._grade_item(item, needs, per_need_scores)
            for item in evidence
        )
        required_need_grades = tuple(grade for grade in need_grades if grade.required)
        coverage_score = (
            sum(grade.coverage_score for grade in required_need_grades) / len(required_need_grades)
            if required_need_grades
            else 1.0
        )

        relevant_count = sum(1 for grade in evidence_grades if grade.relevant)
        if relevant_count == 0:
            status = EvidenceSufficiency.MISSING
            rationale = "No retrieved chunk supports any required information need."
        elif required_need_grades and all(grade.supported for grade in required_need_grades):
            status = EvidenceSufficiency.SUFFICIENT
            rationale = "Every required information need reaches the lexical support threshold."
        else:
            status = EvidenceSufficiency.WEAK
            unresolved = sum(1 for grade in required_need_grades if not grade.supported)
            rationale = f"Evidence is relevant, but {unresolved} required information need(s) remain partial or missing."

        return EvidenceGradingReport(
            status=status,
            coverage_score=round(coverage_score, 4),
            grades=evidence_grades,
            information_need_grades=need_grades,
            rationale=rationale,
            grader_name=self.name,
        )

    def _grade_need(self, need: InformationNeed, scores: dict[int, float]) -> InformationNeedGrade:
        best_score = max(scores.values(), default=0.0)
        supporting_ranks = tuple(
            rank for rank, score in sorted(scores.items()) if score >= self._relevance_threshold
        )
        if not supporting_ranks:
            status = InformationNeedSupport.MISSING
            rationale = "No chunk has enough lexical overlap with this information need."
        elif best_score >= self._sufficiency_threshold:
            status = InformationNeedSupport.SUPPORTED
            rationale = "At least one chunk reaches the lexical support threshold for this information need."
        else:
            status = InformationNeedSupport.PARTIAL
            rationale = "Some relevant terms are present, but coverage is incomplete for this information need."
        return InformationNeedGrade(
            information_need_id=need.need_id,
            description=need.description,
            status=status,
            coverage_score=round(best_score, 4),
            supporting_evidence_ranks=supporting_ranks,
            rationale=rationale,
            required=need.required,
        )

    def _grade_item(
        self,
        item: EvidenceItem,
        needs: tuple[InformationNeed, ...],
        per_need_scores: dict[str, dict[int, float]],
    ) -> EvidenceGrade:
        scores = {need.need_id: per_need_scores[need.need_id][item.rank] for need in needs}
        best_score = max(scores.values(), default=0.0)
        supported_need_ids = tuple(
            need_id for need_id, score in scores.items() if score >= self._relevance_threshold
        )
        relevant = best_score >= self._relevance_threshold
        rationale = (
            f"The chunk lexically supports {len(supported_need_ids)} information need(s)."
            if relevant
            else "The chunk has limited lexical overlap with every information need."
        )
        return EvidenceGrade(
            evidence_rank=item.rank,
            relevance_score=round(best_score, 4),
            relevant=relevant,
            rationale=rationale,
            supports_information_need_ids=supported_need_ids,
        )


def _single_information_need(question: str) -> InformationNeed:
    normalized = " ".join(question.strip().split())
    return InformationNeed(
        need_id="need_1",
        description=normalized,
        retrieval_query=normalized,
    )


def _lexical_coverage(query: str, text: str) -> float:
    query_terms = _content_terms(query)
    evidence_terms = _content_terms(text)
    if not query_terms or not evidence_terms:
        return 0.0
    return min(1.0, len(query_terms & evidence_terms) / len(query_terms))


def _content_terms(value: str) -> set[str]:
    return {token for token in _tokens(value) if len(token) > 1 and token not in _STOP_WORDS}


def _tokens(value: str) -> Iterable[str]:
    return (match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(value))


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
