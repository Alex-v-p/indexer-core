from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.evidence_context import format_constraint_context, format_evidence_for_prompt
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingError,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
    parse_evidence_grading,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "arbitrate_final_evidence.md"
_STATUS_ORDER = {
    InformationNeedSupport.MISSING: 0,
    InformationNeedSupport.PARTIAL: 1,
    InformationNeedSupport.SUPPORTED: 2,
}


class FinalEvidenceArbitrationError(RuntimeError):
    """Raised when the final evidence arbiter cannot produce a valid report."""


class LLMQuestionEvidenceArbitrator:
    """Strict LLM-backed final filter that can preserve or downgrade prior grades."""

    name = "llm_question_level_evidence_arbitrator"

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        fail_open: bool = True,
        relevance_threshold: float = 0.7,
        information_need_support_threshold: float = 0.8,
        max_chars_per_evidence: int = 1_800,
        max_rationale_chars: int = 500,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1.")
        if not 0.0 <= information_need_support_threshold <= 1.0:
            raise ValueError("information_need_support_threshold must be between 0 and 1.")
        if information_need_support_threshold < relevance_threshold:
            raise ValueError("information_need_support_threshold must be at least relevance_threshold.")
        if max_chars_per_evidence <= 0:
            raise ValueError("max_chars_per_evidence must be positive.")
        if max_rationale_chars <= 0:
            raise ValueError("max_rationale_chars must be positive.")
        self._llm_provider = llm_provider
        self._fail_open = fail_open
        self._relevance_threshold = relevance_threshold
        self._information_need_support_threshold = information_need_support_threshold
        self._max_chars_per_evidence = max_chars_per_evidence
        self._max_rationale_chars = max_rationale_chars

    async def arbitrate(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
        prior_grading: EvidenceGradingReport,
        *,
        constraints: RetrievalConstraints | None = None,
    ) -> EvidenceGradingReport:
        if not evidence or not prior_grading.grades or not prior_grading.answerable:
            return prior_grading
        try:
            raw = await self._llm_provider.generate(
                build_final_evidence_arbitration_prompt(
                    question,
                    evidence,
                    information_needs=information_needs,
                    prior_grading=prior_grading,
                    relevance_threshold=self._relevance_threshold,
                    max_chars_per_evidence=self._max_chars_per_evidence,
                    constraints=constraints,
                ),
            )
            proposed = parse_evidence_grading(
                raw,
                evidence=evidence,
                information_needs=information_needs,
                relevance_threshold=self._relevance_threshold,
                information_need_support_threshold=self._information_need_support_threshold,
                grader_name=self.name,
                max_rationale_chars=self._max_rationale_chars,
            )
            return constrain_arbitration_report(prior_grading, proposed, grader_name=self.name)
        except Exception as exc:
            if not self._fail_open:
                if isinstance(exc, (EvidenceGradingError, FinalEvidenceArbitrationError)):
                    raise FinalEvidenceArbitrationError(str(exc)) from exc
                raise FinalEvidenceArbitrationError("Final evidence arbitration failed.") from exc
            return replace(
                prior_grading,
                rationale=(
                    f"Final evidence arbitration failed open; preserved the stricter aggregated prior grades. "
                    f"Cause: {type(exc).__name__}."
                ),
                grader_name=self.name,
                fallback_used=True,
            )


def build_final_evidence_arbitration_prompt(
    question: str,
    evidence: list[EvidenceItem],
    *,
    information_needs: tuple[InformationNeed, ...],
    prior_grading: EvidenceGradingReport,
    relevance_threshold: float = 0.7,
    max_chars_per_evidence: int = 1_800,
    constraints: RetrievalConstraints | None = None,
) -> str:
    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    if not evidence:
        raise ValueError("evidence must not be empty.")
    if not information_needs:
        raise ValueError("information_needs must not be empty.")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1.")
    if max_chars_per_evidence <= 0:
        raise ValueError("max_chars_per_evidence must be positive.")

    effective_constraints = constraints or RetrievalConstraints()
    prior_needs = {grade.information_need_id: grade for grade in prior_grading.information_need_grades}
    prior_information_needs = "\n".join(
        (
            f"- {need.need_id}: status={prior_needs[need.need_id].status.value}; "
            f"coverage={prior_needs[need.need_id].coverage_score:.2f}; "
            f"description={need.description}; "
            f"prior_supporting_ranks={list(prior_needs[need.need_id].supporting_evidence_ranks)}"
        )
        for need in information_needs
        if need.need_id in prior_needs
    )
    prior_grades = {grade.evidence_rank: grade for grade in prior_grading.grades}
    prior_evidence_grades = "\n".join(
        (
            f"- rank {item.rank}: relevant={prior_grades[item.rank].relevant}; "
            f"score={prior_grades[item.rank].relevance_score:.2f}; "
            f"allowed_support_ids={list(prior_grades[item.rank].supports_information_need_ids)}; "
            f"rationale={prior_grades[item.rank].rationale}"
        )
        for item in sorted(evidence, key=lambda candidate: candidate.rank)
        if item.rank in prior_grades
    )
    evidence_block = "\n\n".join(
        format_evidence_for_prompt(
            item,
            constraints=effective_constraints,
            max_chars=max_chars_per_evidence,
        )
        for item in sorted(evidence, key=lambda candidate: candidate.rank)
    )
    return (
        _load_prompt_template()
        .replace("{{ question }}", normalized)
        .replace("{{ relevance_threshold }}", f"{relevance_threshold:.2f}")
        .replace("{{ constraint_context }}", format_constraint_context(effective_constraints))
        .replace("{{ prior_information_needs }}", prior_information_needs)
        .replace("{{ prior_evidence_grades }}", prior_evidence_grades)
        .replace("{{ evidence }}", evidence_block)
        .strip()
    )


def constrain_arbitration_report(
    prior: EvidenceGradingReport,
    proposed: EvidenceGradingReport,
    *,
    grader_name: str,
) -> EvidenceGradingReport:
    """Prevent the final arbiter from approving evidence or support absent from prior grades."""

    prior_by_rank = {grade.evidence_rank: grade for grade in prior.grades}
    proposed_by_rank = {grade.evidence_rank: grade for grade in proposed.grades}
    final_grades: list[EvidenceGrade] = []
    for rank in sorted(prior_by_rank):
        prior_grade = prior_by_rank[rank]
        proposed_grade = proposed_by_rank.get(rank)
        if proposed_grade is None:
            raise FinalEvidenceArbitrationError(f"Arbitration omitted prior evidence rank {rank}.")
        supports = tuple(
            need_id
            for need_id in proposed_grade.supports_information_need_ids
            if need_id in prior_grade.supports_information_need_ids
        )
        relevant = prior_grade.relevant and proposed_grade.relevant and bool(supports)
        score = min(prior_grade.relevance_score, proposed_grade.relevance_score)
        final_grades.append(
            EvidenceGrade(
                evidence_rank=rank,
                relevance_score=score,
                relevant=relevant,
                rationale=(
                    proposed_grade.rationale
                    if relevant
                    else _rejection_rationale(prior_grade, proposed_grade, supports)
                ),
                supports_information_need_ids=supports if relevant else (),
            ),
        )

    final_by_rank = {grade.evidence_rank: grade for grade in final_grades}
    prior_needs = {grade.information_need_id: grade for grade in prior.information_need_grades}
    proposed_needs = {grade.information_need_id: grade for grade in proposed.information_need_grades}
    final_needs: list[InformationNeedGrade] = []
    for need_id, prior_need in prior_needs.items():
        proposed_need = proposed_needs.get(need_id)
        if proposed_need is None:
            raise FinalEvidenceArbitrationError(f"Arbitration omitted information need {need_id!r}.")
        supporting_ranks = tuple(
            rank
            for rank, grade in sorted(final_by_rank.items())
            if grade.relevant and need_id in grade.supports_information_need_ids
        )
        status = _bounded_status(prior_need.status, proposed_need.status, bool(supporting_ranks))
        coverage = min(prior_need.coverage_score, proposed_need.coverage_score) if supporting_ranks else 0.0
        final_needs.append(
            InformationNeedGrade(
                information_need_id=need_id,
                description=prior_need.description,
                status=status,
                coverage_score=coverage,
                supporting_evidence_ranks=supporting_ranks,
                rationale=proposed_need.rationale,
                required=prior_need.required,
            ),
        )

    required = tuple(grade for grade in final_needs if grade.required)
    relevant_count = sum(1 for grade in final_grades if grade.relevant)
    if relevant_count == 0:
        status = EvidenceSufficiency.MISSING
    elif required and all(grade.supported for grade in required):
        status = EvidenceSufficiency.SUFFICIENT
    else:
        status = EvidenceSufficiency.WEAK
    coverage = (
        sum(grade.coverage_score for grade in required) / len(required)
        if required
        else 1.0
    )
    return EvidenceGradingReport(
        status=status,
        coverage_score=round(coverage, 4),
        grades=tuple(final_grades),
        information_need_grades=tuple(final_needs),
        rationale=proposed.rationale,
        grader_name=grader_name,
        fallback_used=prior.fallback_used or proposed.fallback_used,
    )


def _bounded_status(
    prior: InformationNeedSupport,
    proposed: InformationNeedSupport,
    has_supporting_ranks: bool,
) -> InformationNeedSupport:
    if not has_supporting_ranks:
        return InformationNeedSupport.MISSING
    return prior if _STATUS_ORDER[prior] <= _STATUS_ORDER[proposed] else proposed


def _rejection_rationale(
    prior: EvidenceGrade,
    proposed: EvidenceGrade,
    supports: tuple[str, ...],
) -> str:
    if not prior.relevant:
        return "Rejected because no per-information-need grader approved this chunk."
    if not proposed.relevant:
        return proposed.rationale
    if not supports:
        return "Rejected because the proposed support mapping was not approved by the prior graders."
    return "Rejected by final question-level evidence arbitration."


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")
