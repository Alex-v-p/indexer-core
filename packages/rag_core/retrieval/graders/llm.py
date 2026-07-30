from __future__ import annotations

import json
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.ports import StructuredLLMProvider
from packages.rag_core.retrieval.retrievers.base import callable_accepts_parameter
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.graders.base import EvidenceGrader
from packages.rag_core.retrieval.graders.heuristic import HeuristicEvidenceGrader
from packages.rag_core.retrieval.graders.models import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.evidence_context import format_constraint_context, format_evidence_for_prompt
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
from packages.rag_core.structured_output import (
    StructuredOutputError,
    StructuredValidationRule,
    generate_structured_output,
)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "grade_evidence.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class EvidenceGradingError(RuntimeError):
    """Raised when evidence grading cannot produce a complete structured report."""


class _EvidenceGradingInvalidJSONError(EvidenceGradingError):
    pass


class _EvidenceGradingSchemaError(EvidenceGradingError):
    pass


_VALIDATION_RULES = (
    StructuredValidationRule(
        failure_code="invalid_json",
        exception_types=(_EvidenceGradingInvalidJSONError,),
    ),
    StructuredValidationRule(
        failure_code="schema_mismatch",
        exception_types=(_EvidenceGradingSchemaError,),
    ),
    StructuredValidationRule(
        failure_code="semantic_validation_failed",
        exception_types=(EvidenceGradingError,),
    ),
)


def evidence_grading_validation_rules() -> tuple[StructuredValidationRule, ...]:
    return _VALIDATION_RULES


class LLMEvidenceGrader:
    """Provider-neutral claim/aspect grader with deterministic fail-open behavior."""

    name = "llm_information_need_evidence_grader"

    def __init__(
        self,
        *,
        llm_provider: StructuredLLMProvider,
        fallback_grader: EvidenceGrader | None = None,
        fail_open: bool = True,
        relevance_threshold: float = 0.6,
        information_need_support_threshold: float = 0.75,
        max_chars_per_evidence: int = 2_000,
        max_rationale_chars: int = 500,
        max_repair_attempts: int = 1,
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
        if not isinstance(llm_provider, StructuredLLMProvider):
            raise TypeError("llm_provider must support structured generation.")
        if isinstance(max_repair_attempts, bool) or max_repair_attempts not in (0, 1):
            raise ValueError("max_repair_attempts must be 0 or 1.")
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
        self._max_repair_attempts = max_repair_attempts

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
            result = await generate_structured_output(
                provider=self._llm_provider,
                prompt=build_evidence_grading_prompt(
                    normalized,
                    evidence,
                    information_needs=needs,
                    relevance_threshold=self._relevance_threshold,
                    max_chars_per_evidence=self._max_chars_per_evidence,
                    constraints=constraints,
                ),
                response_schema=evidence_grading_response_schema(
                    evidence=evidence,
                    information_needs=needs,
                    max_rationale_chars=self._max_rationale_chars,
                ),
                parser=lambda response: parse_evidence_grading(
                    response,
                    evidence=evidence,
                    information_needs=needs,
                    relevance_threshold=self._relevance_threshold,
                    information_need_support_threshold=self._information_need_support_threshold,
                    grader_name=self.name,
                    max_rationale_chars=self._max_rationale_chars,
                    allow_legacy_response=False,
                ),
                validation_rules=_VALIDATION_RULES,
                max_repair_attempts=self._max_repair_attempts,
            )
            return replace(result.value, structured_output=result.diagnostics)
        except StructuredOutputError as exc:
            if not self._fail_open:
                raise EvidenceGradingError("Evidence grading failed structured validation.") from exc
            fallback = await self._fallback(
                normalized,
                evidence,
                needs,
                constraints=constraints,
            )
            return replace(
                fallback,
                fallback_used=True,
                structured_output=exc.diagnostics,
            )
        except Exception as exc:
            if not self._fail_open:
                if isinstance(exc, EvidenceGradingError):
                    raise
                raise EvidenceGradingError("Evidence grading failed.") from exc

            fallback = await self._fallback(normalized, evidence, needs, constraints=constraints)
            return replace(fallback, fallback_used=True)

    async def _fallback(
        self,
        question: str,
        evidence: list[EvidenceItem],
        needs: tuple[InformationNeed, ...],
        *,
        constraints: RetrievalConstraints | None,
    ) -> EvidenceGradingReport:
        grade_needs = getattr(self._fallback_grader, "grade_information_needs", None)
        if callable(grade_needs):
            if callable_accepts_parameter(grade_needs, "constraints"):
                return await grade_needs(question, evidence, needs, constraints=constraints)
            return await grade_needs(question, evidence, needs)
        grade = self._fallback_grader.grade
        if callable_accepts_parameter(grade, "constraints"):
            return await grade(question, evidence, constraints=constraints)
        return await grade(question, evidence)


def build_evidence_grading_prompt(
    question: str,
    evidence: list[EvidenceItem],
    *,
    information_needs: tuple[InformationNeed, ...] = (),
    relevance_threshold: float = 0.6,
    max_chars_per_evidence: int = 2_000,
    constraints: RetrievalConstraints | None = None,
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
    effective_constraints = constraints or RetrievalConstraints()
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
        .replace("{{ information_needs }}", information_needs_block)
        .replace("{{ constraint_context }}", format_constraint_context(effective_constraints))
        .replace("{{ evidence }}", evidence_block)
        .strip()
    )


def evidence_grading_response_schema(
    *,
    evidence: list[EvidenceItem],
    information_needs: tuple[InformationNeed, ...],
    max_rationale_chars: int = 500,
) -> dict[str, Any]:
    if not evidence:
        raise ValueError("evidence must not be empty.")
    if not information_needs:
        raise ValueError("information_needs must not be empty.")
    if max_rationale_chars <= 0:
        raise ValueError("max_rationale_chars must be positive.")
    ranks = sorted({item.rank for item in evidence})
    need_ids = [need.need_id for need in information_needs]
    rationale_schema = {
        "type": "string",
        "minLength": 1,
        "maxLength": max_rationale_chars,
    }
    return {
        "type": "object",
        "properties": {
            "grades": {
                "type": "array",
                "minItems": len(ranks),
                "maxItems": len(ranks),
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer", "enum": ranks},
                        "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "supports_information_need_ids": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": need_ids},
                        },
                        "rationale": rationale_schema,
                    },
                    "required": [
                        "rank",
                        "relevance_score",
                        "supports_information_need_ids",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            },
            "information_need_grades": {
                "type": "array",
                "minItems": len(need_ids),
                "maxItems": len(need_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "information_need_id": {"type": "string", "enum": need_ids},
                        "status": {
                            "type": "string",
                            "enum": [status.value for status in InformationNeedSupport],
                        },
                        "coverage_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "supporting_ranks": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {"type": "integer", "enum": ranks},
                        },
                        "rationale": rationale_schema,
                    },
                    "required": [
                        "information_need_id",
                        "status",
                        "coverage_score",
                        "supporting_ranks",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            },
            "rationale": rationale_schema,
        },
        "required": ["grades", "information_need_grades", "rationale"],
        "additionalProperties": False,
    }


def parse_evidence_grading(
    raw_response: str,
    *,
    evidence: list[EvidenceItem],
    information_needs: tuple[InformationNeed, ...] = (),
    relevance_threshold: float = 0.6,
    information_need_support_threshold: float = 0.75,
    grader_name: str = LLMEvidenceGrader.name,
    max_rationale_chars: int = 500,
    allow_legacy_response: bool = True,
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
    modern_fields = {"grades", "information_need_grades", "rationale"}
    legacy_fields = {"grades", "sufficiency", "coverage_score", "rationale"}
    payload_fields = set(payload)
    if payload_fields not in (modern_fields, legacy_fields):
        raise _EvidenceGradingSchemaError("Evidence grading response fields do not match the schema.")
    legacy_response = payload_fields == legacy_fields
    if legacy_response and not allow_legacy_response:
        raise _EvidenceGradingSchemaError("Legacy evidence grading responses are not accepted here.")
    expected_ranks = {item.rank for item in evidence}
    expected_need_ids = {need.need_id for need in needs}

    grades = _parse_evidence_grades(
        payload.get("grades"),
        expected_ranks=expected_ranks,
        expected_need_ids=expected_need_ids,
        relevance_threshold=relevance_threshold,
        max_rationale_chars=max_rationale_chars,
        allow_legacy_support_omission=legacy_response,
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
    allow_legacy_support_omission: bool = False,
) -> tuple[EvidenceGrade, ...]:
    if not isinstance(raw_grades, list):
        raise _EvidenceGradingSchemaError("grades must be a JSON array.")

    grades: list[EvidenceGrade] = []
    seen_ranks: set[int] = set()
    for raw_grade in raw_grades:
        if not isinstance(raw_grade, dict):
            raise _EvidenceGradingSchemaError("Each evidence grade must be a JSON object.")
        expected_fields = {
            "rank",
            "relevance_score",
            "supports_information_need_ids",
            "rationale",
        }
        legacy_fields = expected_fields - {"supports_information_need_ids"}
        if set(raw_grade) != expected_fields and not (
            allow_legacy_support_omission and set(raw_grade) == legacy_fields
        ):
            raise _EvidenceGradingSchemaError("Evidence grade fields do not match the schema.")
        rank = _parse_rank(raw_grade.get("rank"))
        if rank not in expected_ranks:
            raise EvidenceGradingError(f"Evidence grade references unknown rank {rank}.")
        if rank in seen_ranks:
            raise _EvidenceGradingSchemaError(f"Evidence rank {rank} was graded more than once.")
        seen_ranks.add(rank)

        score = _parse_probability(raw_grade.get("relevance_score"), "relevance_score")
        if allow_legacy_support_omission and "supports_information_need_ids" not in raw_grade:
            supported_ids = tuple(expected_need_ids) if score >= relevance_threshold else ()
        else:
            supported_ids = _parse_string_list(
                raw_grade.get("supports_information_need_ids"),
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
        raise _EvidenceGradingSchemaError("information_need_grades must be a JSON array.")

    needs_by_id = {need.need_id: need for need in needs}
    grades_by_rank = {grade.evidence_rank: grade for grade in grades}
    parsed: list[InformationNeedGrade] = []
    seen_ids: set[str] = set()
    for raw_grade in raw_need_grades:
        if not isinstance(raw_grade, dict):
            raise _EvidenceGradingSchemaError("Each information-need grade must be a JSON object.")
        if set(raw_grade) != {
            "information_need_id",
            "status",
            "coverage_score",
            "supporting_ranks",
            "rationale",
        }:
            raise _EvidenceGradingSchemaError("Information-need grade fields do not match the schema.")
        need_id = raw_grade.get("information_need_id")
        if not isinstance(need_id, str):
            raise _EvidenceGradingSchemaError("information_need_id must be a string.")
        if need_id not in needs_by_id:
            raise EvidenceGradingError(f"Unknown information_need_id {need_id!r}.")
        if need_id in seen_ids:
            raise _EvidenceGradingSchemaError(f"Information need {need_id!r} was graded more than once.")
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

        raw_status = raw_grade.get("status")
        if not isinstance(raw_status, str):
            raise _EvidenceGradingSchemaError("Information-need status must be a string.")
        try:
            requested_status = InformationNeedSupport(raw_status)
        except ValueError as exc:
            raise _EvidenceGradingSchemaError("Information-need status is missing or invalid.") from exc
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
    raw_sufficiency = payload.get("sufficiency")
    if not isinstance(raw_sufficiency, str):
        raise _EvidenceGradingSchemaError("sufficiency must be a string.")
    try:
        requested_sufficiency = EvidenceSufficiency(raw_sufficiency)
    except ValueError as exc:
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
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise _EvidenceGradingInvalidJSONError("Evidence grading response contained invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise _EvidenceGradingSchemaError("Evidence grading response must be a JSON object.")
    return parsed


def _parse_rank(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _EvidenceGradingSchemaError("Each grade rank must be a positive integer.")
    return value


def _parse_rank_list(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise _EvidenceGradingSchemaError("supporting_ranks must be a JSON array.")
    ranks = tuple(_parse_rank(item) for item in value)
    if len(ranks) != len(set(ranks)):
        raise _EvidenceGradingSchemaError("supporting_ranks must not contain duplicates.")
    return ranks


def _parse_string_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _EvidenceGradingSchemaError(f"{field_name} must be a JSON array.")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _EvidenceGradingSchemaError(f"{field_name} may only contain non-empty strings.")
        if item in parsed:
            raise _EvidenceGradingSchemaError(f"{field_name} must not contain duplicates.")
        parsed.append(item)
    return tuple(parsed)


def _parse_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _EvidenceGradingSchemaError(f"{field_name} must be a number between 0 and 1.")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise _EvidenceGradingSchemaError(f"{field_name} must be between 0 and 1.")
    return parsed


def _normalize_rationale(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        raise _EvidenceGradingSchemaError("Evidence grading rationales must be strings.")
    if len(value) > max_chars:
        raise _EvidenceGradingSchemaError("Evidence grading rationale exceeds the character limit.")
    rationale = " ".join(value.strip().split())
    if not rationale:
        raise _EvidenceGradingSchemaError("Evidence grading rationales must not be empty.")
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
