from __future__ import annotations

import json
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.planning import InformationNeed
from packages.rag_core.retrieval.graders.base import EvidenceGrader
from packages.rag_core.retrieval.graders.heuristic import HeuristicEvidenceGrader
from packages.rag_core.retrieval.graders.models import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "grade_evidence.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class EvidenceGradingError(RuntimeError):
    """Raised when evidence grading cannot produce a complete structured report."""


class LLMEvidenceGrader:
    """Provider-neutral claim/aspect grader with deterministic fail-open behavior."""

    name = "llm_information_need_evidence_grader"

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        fallback_grader: EvidenceGrader | None = None,
        fail_open: bool = True,
        relevance_threshold: float = 0.6,
        information_need_support_threshold: float = 0.75,
        max_chars_per_evidence: int = 2_000,
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
        self._fallback_grader = fallback_grader or HeuristicEvidenceGrader(
            relevance_threshold=min(relevance_threshold, 0.35),
            sufficiency_threshold=max(relevance_threshold, 0.6),
        )
        self._fail_open = fail_open
        self._relevance_threshold = relevance_threshold
        self._information_need_support_threshold = information_need_support_threshold
        self._max_chars_per_evidence = max_chars_per_evidence
        self._max_rationale_chars = max_rationale_chars

    async def grade(self, question: str, evidence: list[EvidenceItem]) -> EvidenceGradingReport:
        return await self.grade_information_needs(
            question,
            evidence,
            (_single_information_need(question),),
        )

    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
    ) -> EvidenceGradingReport:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")
        needs = information_needs or (_single_information_need(normalized),)
        if not evidence:
            return EvidenceGradingReport(
                status=EvidenceSufficiency.MISSING,
                coverage_score=0.0,
                grades=(),
                information_need_grades=tuple(
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
                ),
                rationale="No evidence was retrieved for the question.",
                grader_name=self.name,
            )

        try:
            response = await self._llm_provider.generate(
                build_evidence_grading_prompt(
                    normalized,
                    evidence,
                    information_needs=needs,
                    relevance_threshold=self._relevance_threshold,
                    max_chars_per_evidence=self._max_chars_per_evidence,
                ),
            )
            return parse_evidence_grading(
                response,
                evidence=evidence,
                information_needs=needs,
                relevance_threshold=self._relevance_threshold,
                information_need_support_threshold=self._information_need_support_threshold,
                grader_name=self.name,
                max_rationale_chars=self._max_rationale_chars,
            )
        except Exception as exc:
            if not self._fail_open:
                if isinstance(exc, EvidenceGradingError):
                    raise
                raise EvidenceGradingError("Evidence grading failed.") from exc

            grade_needs = getattr(self._fallback_grader, "grade_information_needs", None)
            if callable(grade_needs):
                fallback = await grade_needs(normalized, evidence, needs)
            else:
                fallback = await self._fallback_grader.grade(normalized, evidence)
            return replace(fallback, fallback_used=True)


def build_evidence_grading_prompt(
    question: str,
    evidence: list[EvidenceItem],
    *,
    information_needs: tuple[InformationNeed, ...] = (),
    relevance_threshold: float = 0.6,
    max_chars_per_evidence: int = 2_000,
) -> str:
    """Build a claim/aspect-level grading prompt from the checked-in template."""

    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    if not evidence:
        raise ValueError("evidence must not be empty.")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1.")
    if max_chars_per_evidence <= 0:
        raise ValueError("max_chars_per_evidence must be positive.")

    needs = information_needs or (_single_information_need(normalized),)
    information_needs_block = "\n".join(
        f"- {need.need_id}: {need.description}\n  Retrieval query: {need.retrieval_query}"
        for need in needs
    )
    evidence_block = "\n\n".join(
        f"[{item.rank}]\n{item.text.strip()[:max_chars_per_evidence]}"
        for item in sorted(evidence, key=lambda candidate: candidate.rank)
    )
    return (
        _load_prompt_template()
        .replace("{{ question }}", normalized)
        .replace("{{ relevance_threshold }}", f"{relevance_threshold:.2f}")
        .replace("{{ information_needs }}", information_needs_block)
        .replace("{{ evidence }}", evidence_block)
        .strip()
    )


def parse_evidence_grading(
    raw_response: str,
    *,
    evidence: list[EvidenceItem],
    information_needs: tuple[InformationNeed, ...] = (),
    relevance_threshold: float = 0.6,
    information_need_support_threshold: float = 0.75,
    grader_name: str = LLMEvidenceGrader.name,
    max_rationale_chars: int = 500,
) -> EvidenceGradingReport:
    """Parse complete chunk and information-need coverage from one LLM response."""

    if not evidence:
        raise ValueError("evidence must not be empty.")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1.")
    if not 0.0 <= information_need_support_threshold <= 1.0:
        raise ValueError("information_need_support_threshold must be between 0 and 1.")
    if information_need_support_threshold < relevance_threshold:
        raise ValueError("information_need_support_threshold must be at least relevance_threshold.")
    if max_rationale_chars <= 0:
        raise ValueError("max_rationale_chars must be positive.")

    needs = information_needs or (
        InformationNeed(
            need_id="need_1",
            description="Answer the submitted question.",
            retrieval_query="Answer the submitted question.",
        ),
    )
    payload = _extract_json_object(raw_response)
    expected_ranks = {item.rank for item in evidence}
    expected_need_ids = {need.need_id for need in needs}

    grades = _parse_evidence_grades(
        payload.get("grades"),
        expected_ranks=expected_ranks,
        expected_need_ids=expected_need_ids,
        relevance_threshold=relevance_threshold,
        max_rationale_chars=max_rationale_chars,
    )

    raw_need_grades = payload.get("information_need_grades")
    if raw_need_grades is None:
        need_grades = _parse_legacy_need_grade(
            payload,
            needs=needs,
            grades=grades,
            information_need_support_threshold=information_need_support_threshold,
            max_rationale_chars=max_rationale_chars,
        )
    else:
        need_grades = _parse_information_need_grades(
            raw_need_grades,
            needs=needs,
            grades=grades,
            information_need_support_threshold=information_need_support_threshold,
            max_rationale_chars=max_rationale_chars,
        )

    relevant_count = sum(1 for grade in grades if grade.relevant)
    required_need_grades = tuple(grade for grade in need_grades if grade.required)
    if relevant_count == 0:
        status = EvidenceSufficiency.MISSING
    elif required_need_grades and all(grade.supported for grade in required_need_grades):
        status = EvidenceSufficiency.SUFFICIENT
    else:
        status = EvidenceSufficiency.WEAK

    coverage_score = (
        sum(grade.coverage_score for grade in required_need_grades) / len(required_need_grades)
        if required_need_grades
        else 1.0
    )
    return EvidenceGradingReport(
        status=status,
        coverage_score=round(coverage_score, 4),
        grades=tuple(sorted(grades, key=lambda grade: grade.evidence_rank)),
        information_need_grades=tuple(
            sorted(need_grades, key=lambda grade: _need_order(grade.information_need_id, needs))
        ),
        rationale=_normalize_rationale(payload.get("rationale"), max_rationale_chars),
        grader_name=grader_name,
    )


def _parse_evidence_grades(
    raw_grades: object,
    *,
    expected_ranks: set[int],
    expected_need_ids: set[str],
    relevance_threshold: float,
    max_rationale_chars: int,
) -> tuple[EvidenceGrade, ...]:
    if not isinstance(raw_grades, list):
        raise EvidenceGradingError("grades must be a JSON array.")

    grades: list[EvidenceGrade] = []
    seen_ranks: set[int] = set()
    for raw_grade in raw_grades:
        if not isinstance(raw_grade, dict):
            raise EvidenceGradingError("Each evidence grade must be a JSON object.")
        rank = _parse_rank(raw_grade.get("rank"))
        if rank not in expected_ranks:
            raise EvidenceGradingError(f"Evidence grade references unknown rank {rank}.")
        if rank in seen_ranks:
            raise EvidenceGradingError(f"Evidence rank {rank} was graded more than once.")
        seen_ranks.add(rank)

        score = _parse_probability(raw_grade.get("relevance_score"), "relevance_score")
        supported_ids = _parse_string_list(
            raw_grade.get("supports_information_need_ids", []),
            field_name="supports_information_need_ids",
        )
        unknown_ids = set(supported_ids) - expected_need_ids
        if unknown_ids:
            rendered = ", ".join(sorted(unknown_ids))
            raise EvidenceGradingError(f"Evidence grade references unknown information need ids: {rendered}.")
        if supported_ids and score < relevance_threshold:
            raise EvidenceGradingError(
                f"Evidence rank {rank} supports information needs but is below the relevance threshold.",
            )
        grades.append(
            EvidenceGrade(
                evidence_rank=rank,
                relevance_score=score,
                relevant=score >= relevance_threshold,
                rationale=_normalize_rationale(raw_grade.get("rationale"), max_rationale_chars),
                supports_information_need_ids=supported_ids,
            ),
        )

    missing_ranks = expected_ranks - seen_ranks
    if missing_ranks:
        rendered = ", ".join(str(rank) for rank in sorted(missing_ranks))
        raise EvidenceGradingError(f"Evidence grading response omitted ranks: {rendered}.")
    return tuple(grades)


def _parse_information_need_grades(
    raw_need_grades: object,
    *,
    needs: tuple[InformationNeed, ...],
    grades: tuple[EvidenceGrade, ...],
    information_need_support_threshold: float,
    max_rationale_chars: int,
) -> tuple[InformationNeedGrade, ...]:
    if not isinstance(raw_need_grades, list):
        raise EvidenceGradingError("information_need_grades must be a JSON array.")

    needs_by_id = {need.need_id: need for need in needs}
    grades_by_rank = {grade.evidence_rank: grade for grade in grades}
    parsed: list[InformationNeedGrade] = []
    seen_ids: set[str] = set()
    for raw_grade in raw_need_grades:
        if not isinstance(raw_grade, dict):
            raise EvidenceGradingError("Each information-need grade must be a JSON object.")
        need_id = str(raw_grade.get("information_need_id") or "").strip()
        if need_id not in needs_by_id:
            raise EvidenceGradingError(f"Unknown information_need_id {need_id!r}.")
        if need_id in seen_ids:
            raise EvidenceGradingError(f"Information need {need_id!r} was graded more than once.")
        seen_ids.add(need_id)

        coverage_score = _parse_probability(raw_grade.get("coverage_score"), "coverage_score")
        supporting_ranks = _parse_rank_list(raw_grade.get("supporting_ranks", []))
        unknown_ranks = set(supporting_ranks) - set(grades_by_rank)
        if unknown_ranks:
            rendered = ", ".join(str(rank) for rank in sorted(unknown_ranks))
            raise EvidenceGradingError(f"Information need {need_id!r} references unknown ranks: {rendered}.")
        for rank in supporting_ranks:
            if need_id not in grades_by_rank[rank].supports_information_need_ids:
                raise EvidenceGradingError(
                    f"Information need {need_id!r} lists rank {rank}, but that chunk does not list the need as supported.",
                )
        reciprocal_ranks = {
            grade.evidence_rank for grade in grades if need_id in grade.supports_information_need_ids
        }
        if reciprocal_ranks != set(supporting_ranks):
            raise EvidenceGradingError(
                f"Information need {need_id!r} and chunk support mappings are inconsistent.",
            )

        try:
            requested_status = InformationNeedSupport(str(raw_grade.get("status")).strip().casefold())
        except ValueError as exc:
            raise EvidenceGradingError("Information-need status is missing or invalid.") from exc
        normalized_status = _normalize_need_status(
            requested_status,
            supporting_ranks=supporting_ranks,
            coverage_score=coverage_score,
            support_threshold=information_need_support_threshold,
        )
        need = needs_by_id[need_id]
        parsed.append(
            InformationNeedGrade(
                information_need_id=need_id,
                description=need.description,
                status=normalized_status,
                coverage_score=coverage_score,
                supporting_evidence_ranks=supporting_ranks,
                rationale=_normalize_rationale(raw_grade.get("rationale"), max_rationale_chars),
                required=need.required,
            ),
        )

    missing_ids = set(needs_by_id) - seen_ids
    if missing_ids:
        rendered = ", ".join(sorted(missing_ids))
        raise EvidenceGradingError(f"Evidence grading response omitted information need ids: {rendered}.")
    return tuple(parsed)


def _parse_legacy_need_grade(
    payload: dict[str, Any],
    *,
    needs: tuple[InformationNeed, ...],
    grades: tuple[EvidenceGrade, ...],
    information_need_support_threshold: float,
    max_rationale_chars: int,
) -> tuple[InformationNeedGrade, ...]:
    if len(needs) != 1:
        raise EvidenceGradingError("information_need_grades are required for decomposed questions.")
    try:
        requested_sufficiency = EvidenceSufficiency(str(payload["sufficiency"]).strip().casefold())
    except (KeyError, ValueError) as exc:
        raise EvidenceGradingError("information_need_grades are missing.") from exc

    coverage_score = _parse_probability(payload.get("coverage_score"), "coverage_score")
    supporting_ranks = tuple(grade.evidence_rank for grade in grades if grade.relevant)
    requested_status = {
        EvidenceSufficiency.MISSING: InformationNeedSupport.MISSING,
        EvidenceSufficiency.WEAK: InformationNeedSupport.PARTIAL,
        EvidenceSufficiency.SUFFICIENT: InformationNeedSupport.SUPPORTED,
    }[requested_sufficiency]
    status = _normalize_need_status(
        requested_status,
        supporting_ranks=supporting_ranks,
        coverage_score=coverage_score,
        support_threshold=information_need_support_threshold,
    )
    need = needs[0]
    return (
        InformationNeedGrade(
            information_need_id=need.need_id,
            description=need.description,
            status=status,
            coverage_score=coverage_score,
            supporting_evidence_ranks=supporting_ranks,
            rationale=_normalize_rationale(payload.get("rationale"), max_rationale_chars),
            required=need.required,
        ),
    )


def _normalize_need_status(
    requested_status: InformationNeedSupport,
    *,
    supporting_ranks: tuple[int, ...],
    coverage_score: float,
    support_threshold: float,
) -> InformationNeedSupport:
    if not supporting_ranks:
        return InformationNeedSupport.MISSING
    if requested_status is InformationNeedSupport.SUPPORTED and coverage_score >= support_threshold:
        return InformationNeedSupport.SUPPORTED
    return InformationNeedSupport.PARTIAL


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw_response.strip())
    start = cleaned.find("{")
    if start < 0:
        raise EvidenceGradingError("Evidence grading response did not contain a JSON object.")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise EvidenceGradingError("Evidence grading response contained invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise EvidenceGradingError("Evidence grading response must be a JSON object.")
    return parsed


def _parse_rank(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceGradingError("Each grade rank must be a positive integer.")
    return value


def _parse_rank_list(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise EvidenceGradingError("supporting_ranks must be a JSON array.")
    ranks = tuple(_parse_rank(item) for item in value)
    if len(ranks) != len(set(ranks)):
        raise EvidenceGradingError("supporting_ranks must not contain duplicates.")
    return ranks


def _parse_string_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvidenceGradingError(f"{field_name} must be a JSON array.")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise EvidenceGradingError(f"{field_name} may only contain non-empty strings.")
        normalized = item.strip()
        if normalized not in parsed:
            parsed.append(normalized)
    return tuple(parsed)


def _parse_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceGradingError(f"{field_name} must be a number between 0 and 1.")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise EvidenceGradingError(f"{field_name} must be between 0 and 1.")
    return parsed


def _normalize_rationale(value: object, max_chars: int) -> str:
    rationale = " ".join(str(value or "").strip().split())[:max_chars].strip()
    if not rationale:
        raise EvidenceGradingError("Evidence grading rationales must not be empty.")
    return rationale


def _single_information_need(question: str) -> InformationNeed:
    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    return InformationNeed(
        need_id="need_1",
        description=normalized,
        retrieval_query=normalized,
    )


def _need_order(need_id: str, needs: tuple[InformationNeed, ...]) -> int:
    return next(index for index, need in enumerate(needs) if need.need_id == need_id)
