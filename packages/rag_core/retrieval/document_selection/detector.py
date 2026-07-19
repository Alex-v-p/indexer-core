from __future__ import annotations

import re
from dataclasses import dataclass, field

from packages.rag_core.documents.preferences import (
    DocumentPreference,
    DocumentReference,
    document_reference_from_values,
    merge_document_references,
)
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.graders import EvidenceGradingReport, InformationNeedSupport
from packages.rag_core.retrieval.models import EvidenceItem

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_GENERIC_DOCUMENT_TOKENS = frozenset(
    {
        "document",
        "documents",
        "file",
        "files",
        "pdf",
        "project",
        "plan",
        "spec",
        "specification",
        "functional",
        "realization",
        "realisation",
        "draft",
        "report",
        "version",
        "final",
        "the",
        "a",
        "an",
        "for",
        "of",
        "and",
    },
)


@dataclass(slots=True)
class _DocumentScore:
    reference: DocumentReference
    evidence_ranks: list[int] = field(default_factory=list)
    relevance_scores: list[float] = field(default_factory=list)
    name_overlap_scores: list[float] = field(default_factory=list)

    @property
    def score(self) -> float:
        strongest_relevance = max(self.relevance_scores, default=0.0)
        strongest_name_overlap = max(self.name_overlap_scores, default=0.0)
        breadth_bonus = min(0.08, max(0, len(self.evidence_ranks) - 1) * 0.04)
        return min(1.0, 0.74 * strongest_relevance + 0.21 * strongest_name_overlap + breadth_bonus)


class RuleBasedPrimaryDocumentDetector:
    """Learn a stable soft document preference from direct per-need evidence."""

    name = "rule_based_soft_primary_document_detector"

    def __init__(
        self,
        *,
        min_score: float = 0.65,
        min_margin: float = 0.08,
        replacement_margin: float = 0.12,
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1.")
        if not 0.0 <= min_margin <= 1.0:
            raise ValueError("min_margin must be between 0 and 1.")
        if not 0.0 <= replacement_margin <= 1.0:
            raise ValueError("replacement_margin must be between 0 and 1.")
        self._min_score = min_score
        self._min_margin = min_margin
        self._replacement_margin = replacement_margin

    def detect(
        self,
        *,
        question: str,
        information_need: InformationNeed,
        evidence: list[EvidenceItem],
        grading: EvidenceGradingReport,
        existing_preference: DocumentPreference | None = None,
    ) -> DocumentPreference | None:
        grade_by_rank = {
            grade.evidence_rank: grade
            for grade in grading.grades
            if grade.relevant and information_need.need_id in grade.supports_information_need_ids
        }
        if not grade_by_rank:
            return existing_preference

        query_tokens = _informative_tokens(
            " ".join((question, information_need.description, information_need.retrieval_query)),
        )
        candidates: dict[str, _DocumentScore] = {}
        for item in evidence:
            grade = grade_by_rank.get(item.rank)
            if grade is None:
                continue
            reference = document_reference_from_values(
                rank=item.rank,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                metadata=item.metadata,
            )
            candidate = candidates.get(reference.key)
            if candidate is None:
                candidate = _DocumentScore(reference=reference)
                candidates[reference.key] = candidate
            else:
                candidate.reference = merge_document_references(candidate.reference, reference)
            candidate.evidence_ranks.append(item.rank)
            candidate.relevance_scores.append(grade.relevance_score)
            candidate.name_overlap_scores.append(_document_name_overlap(reference, query_tokens))

        ranked = sorted(candidates.values(), key=lambda candidate: candidate.score, reverse=True)
        if not ranked:
            return existing_preference
        best = ranked[0]
        runner_up_score = ranked[1].score if len(ranked) > 1 else 0.0
        margin = max(0.0, best.score - runner_up_score)
        if best.score < self._min_score or (len(ranked) > 1 and margin < self._min_margin):
            return existing_preference

        need_grade = next(
            (
                grade
                for grade in grading.information_need_grades
                if grade.information_need_id == information_need.need_id
            ),
            None,
        )
        if need_grade is None or need_grade.status is InformationNeedSupport.MISSING:
            return existing_preference
        support_factor = {
            InformationNeedSupport.SUPPORTED: 1.0,
            InformationNeedSupport.PARTIAL: 0.8,
        }[need_grade.status]
        confidence = min(1.0, (best.score * 0.75 + min(0.25, margin)) * support_factor)
        proposed = DocumentPreference(
            document=best.reference,
            score=round(best.score, 4),
            confidence=round(confidence, 4),
            margin=round(margin, 4),
            supporting_information_need_ids=(information_need.need_id,),
            supporting_evidence_ranks=tuple(dict.fromkeys(best.evidence_ranks)),
            rationale=(
                f"{best.reference.display_name!r} was the strongest directly graded document for "
                f"{information_need.need_id}, scoring {best.score:.2f} with a {margin:.2f} lead. "
                "This is a soft retrieval preference and does not exclude other documents."
            ),
            detector_name=self.name,
        )
        return self._merge_or_preserve(existing_preference, proposed)

    def _merge_or_preserve(
        self,
        existing: DocumentPreference | None,
        proposed: DocumentPreference,
    ) -> DocumentPreference:
        if existing is None:
            return proposed
        if existing.document.key == proposed.document.key:
            return DocumentPreference(
                document=merge_document_references(existing.document, proposed.document),
                score=max(existing.score, proposed.score),
                confidence=max(existing.confidence, proposed.confidence),
                margin=max(existing.margin, proposed.margin),
                supporting_information_need_ids=tuple(
                    dict.fromkeys(
                        (*existing.supporting_information_need_ids, *proposed.supporting_information_need_ids),
                    ),
                ),
                supporting_evidence_ranks=tuple(
                    dict.fromkeys((*existing.supporting_evidence_ranks, *proposed.supporting_evidence_ranks)),
                ),
                rationale=(
                    f"The existing preference for {existing.document.display_name!r} was reinforced by "
                    f"additional directly graded evidence."
                ),
                detector_name=self.name,
            )
        if proposed.score >= existing.score + self._replacement_margin:
            return proposed
        return existing


class NoPrimaryDocumentDetector:
    """Compatibility detector used when soft document preference is disabled."""

    name = "disabled_primary_document_detector"

    def detect(
        self,
        *,
        question: str,
        information_need: InformationNeed,
        evidence: list[EvidenceItem],
        grading: EvidenceGradingReport,
        existing_preference: DocumentPreference | None = None,
    ) -> DocumentPreference | None:
        del question, information_need, evidence, grading
        return existing_preference


def _document_name_overlap(reference: DocumentReference, query_tokens: set[str]) -> float:
    document_tokens = {
        token
        for name in (*reference.normalized_names, reference.display_name)
        for token in _informative_tokens(name)
    }
    if not document_tokens or not query_tokens:
        return 0.0
    overlap = document_tokens.intersection(query_tokens)
    if not overlap:
        return 0.0
    return min(1.0, 0.6 + 0.2 * len(overlap))


def _informative_tokens(value: str) -> set[str]:
    expanded = _CAMEL_BOUNDARY_PATTERN.sub(" ", value)
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(expanded)
        if len(token) >= 2 and token.casefold() not in _GENERIC_DOCUMENT_TOKENS
    }
