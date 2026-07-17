from __future__ import annotations

import json
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.graders.base import EvidenceGrader
from packages.rag_core.retrieval.graders.heuristic import HeuristicEvidenceGrader
from packages.rag_core.retrieval.graders.models import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
)
from packages.rag_core.retrieval.models import EvidenceItem

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "grade_evidence.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class EvidenceGradingError(RuntimeError):
    """Raised when evidence grading cannot produce a complete structured report."""


class LLMEvidenceGrader:
    """Provider-neutral LLM grader with deterministic fail-open behavior."""

    name = "llm_evidence_grader"

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        fallback_grader: EvidenceGrader | None = None,
        fail_open: bool = True,
        relevance_threshold: float = 0.6,
        max_chars_per_evidence: int = 2_000,
        max_rationale_chars: int = 500,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1.")
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
        self._max_chars_per_evidence = max_chars_per_evidence
        self._max_rationale_chars = max_rationale_chars

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

        try:
            response = await self._llm_provider.generate(
                build_evidence_grading_prompt(
                    normalized,
                    evidence,
                    relevance_threshold=self._relevance_threshold,
                    max_chars_per_evidence=self._max_chars_per_evidence,
                ),
            )
            return parse_evidence_grading(
                response,
                evidence=evidence,
                relevance_threshold=self._relevance_threshold,
                grader_name=self.name,
                max_rationale_chars=self._max_rationale_chars,
            )
        except Exception as exc:
            if not self._fail_open:
                if isinstance(exc, EvidenceGradingError):
                    raise
                raise EvidenceGradingError("Evidence grading failed.") from exc

            fallback = await self._fallback_grader.grade(normalized, evidence)
            return replace(fallback, fallback_used=True)


def build_evidence_grading_prompt(
    question: str,
    evidence: list[EvidenceItem],
    *,
    relevance_threshold: float = 0.6,
    max_chars_per_evidence: int = 2_000,
) -> str:
    """Build the grading prompt from the checked-in markdown template."""

    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    if not evidence:
        raise ValueError("evidence must not be empty.")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1.")
    if max_chars_per_evidence <= 0:
        raise ValueError("max_chars_per_evidence must be positive.")

    evidence_block = "\n\n".join(
        f"[{item.rank}]\n{item.text.strip()[:max_chars_per_evidence]}"
        for item in sorted(evidence, key=lambda candidate: candidate.rank)
    )
    return (
        _load_prompt_template()
        .replace("{{ question }}", normalized)
        .replace("{{ relevance_threshold }}", f"{relevance_threshold:.2f}")
        .replace("{{ evidence }}", evidence_block)
        .strip()
    )


def parse_evidence_grading(
    raw_response: str,
    *,
    evidence: list[EvidenceItem],
    relevance_threshold: float = 0.6,
    grader_name: str = LLMEvidenceGrader.name,
    max_rationale_chars: int = 500,
) -> EvidenceGradingReport:
    """Parse a complete per-rank grading response and normalize its invariants."""

    if not evidence:
        raise ValueError("evidence must not be empty.")
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1.")
    if max_rationale_chars <= 0:
        raise ValueError("max_rationale_chars must be positive.")

    payload = _extract_json_object(raw_response)
    expected_ranks = {item.rank for item in evidence}
    raw_grades = payload.get("grades")
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
        rationale = _normalize_rationale(raw_grade.get("rationale"), max_rationale_chars)
        grades.append(
            EvidenceGrade(
                evidence_rank=rank,
                relevance_score=score,
                relevant=score >= relevance_threshold,
                rationale=rationale,
            ),
        )

    missing_ranks = expected_ranks - seen_ranks
    if missing_ranks:
        rendered = ", ".join(str(rank) for rank in sorted(missing_ranks))
        raise EvidenceGradingError(f"Evidence grading response omitted ranks: {rendered}.")

    try:
        requested_status = EvidenceSufficiency(str(payload["sufficiency"]).strip().casefold())
    except (KeyError, ValueError) as exc:
        raise EvidenceGradingError("sufficiency is missing or invalid.") from exc

    relevant_count = sum(1 for grade in grades if grade.relevant)
    if relevant_count == 0:
        status = EvidenceSufficiency.MISSING
    elif requested_status is EvidenceSufficiency.MISSING:
        status = EvidenceSufficiency.WEAK
    else:
        status = requested_status

    return EvidenceGradingReport(
        status=status,
        coverage_score=_parse_probability(payload.get("coverage_score"), "coverage_score"),
        grades=tuple(sorted(grades, key=lambda grade: grade.evidence_rank)),
        rationale=_normalize_rationale(payload.get("rationale"), max_rationale_chars),
        grader_name=grader_name,
    )


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
