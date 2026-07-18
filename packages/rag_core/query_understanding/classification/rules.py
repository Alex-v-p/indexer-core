from __future__ import annotations

import re

from packages.rag_core.query_understanding.classification.models import MetadataFilterHint, QueryClassification, QueryType
from packages.rag_core.query_understanding.versioning import detect_document_version_constraint
from packages.rag_core.query_understanding.temporal import detect_document_date_constraints

_COMPARISON_PATTERN = re.compile(
    r"\b(compare|comparison|contrast|difference|differences|different from|similarities|similarity|versus|vs\.?|better than|worse than)\b",
    re.IGNORECASE,
)
_VERSION_PATTERN = re.compile(
    r"\b(latest|newest|current|previous|prior|older|oldest|version|revision|release|edition|as of|historical)\b|\bv\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_DATE_RANGE_PATTERN = re.compile(
    r"\b(before|after|between|during|until|since|as of)\b|\bfrom\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_BROAD_PATTERN = re.compile(
    r"\b(explain|overview|summari[sz]e|describe|discuss|walk me through|why|implications?|architecture|process)\b|\bhow (?:does|do|did|can|should|would)\b",
    re.IGNORECASE,
)
_SECTION_PATTERN = re.compile(r"\b(page|pages|section|chapter|appendix|paragraph)\b", re.IGNORECASE)
_FILE_TYPE_PATTERN = re.compile(r"\b(pdf|markdown|md|docx?|text file|txt)\b", re.IGNORECASE)
_AUTHOR_PATTERN = re.compile(r"\b(author|written by|created by|published by)\b", re.IGNORECASE)
_DOCUMENT_PATTERN = re.compile(
    r"\b(?:in|from|according to)\s+(?:the\s+)?(?:[\w.-]+\s+){0,4}(document|report|manual|policy|proposal|specification|spec|paper|file)\b",
    re.IGNORECASE,
)
_FILENAME_PATTERN = re.compile(r"\b[\w-]+\.(?:pdf|md|markdown|txt|docx?)\b", re.IGNORECASE)


class HeuristicQueryClassifier:
    """Deterministic classifier used directly in tests and as an LLM fallback."""

    name = "heuristic_rules"

    def __init__(self, *, timezone_name: str = "UTC") -> None:
        self._timezone_name = timezone_name

    async def classify(self, question: str) -> QueryClassification:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")
        return classify_query_heuristically(normalized, timezone_name=self._timezone_name)


def classify_query_heuristically(
    question: str,
    *,
    timezone_name: str = "UTC",
) -> QueryClassification:
    """Classify a normalized question using conservative, explainable rules."""

    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")

    has_comparison = _COMPARISON_PATTERN.search(normalized) is not None
    has_version = _VERSION_PATTERN.search(normalized) is not None or _YEAR_PATTERN.search(normalized) is not None

    if has_comparison:
        query_type = QueryType.COMPARISON
        rationale = "The question explicitly asks to compare or contrast multiple subjects."
        confidence = 0.88
    elif has_version:
        query_type = QueryType.VERSION_SPECIFIC
        rationale = "The answer depends on a named, relative, dated, or historical version."
        confidence = 0.86
    elif _BROAD_PATTERN.search(normalized):
        query_type = QueryType.BROAD_EXPLANATION
        rationale = "The question requests an explanation, summary, process, or broader interpretation."
        confidence = 0.78
    else:
        query_type = QueryType.FACTUAL_LOOKUP
        rationale = "The question appears to request a focused fact or directly retrievable detail."
        confidence = 0.66

    version_constraint = detect_document_version_constraint(normalized)
    date_constraints = detect_document_date_constraints(normalized, timezone_name=timezone_name)
    hints = _metadata_filter_hints(
        normalized,
        has_version=has_version or version_constraint.active,
        has_dates=bool(date_constraints),
    )
    return QueryClassification(
        query_type=query_type,
        confidence=confidence,
        needs_metadata_filters=bool(hints),
        metadata_filter_hints=hints,
        rationale=rationale,
        classifier_name=HeuristicQueryClassifier.name,
        version_constraint=version_constraint,
        date_constraints=date_constraints,
    )


def _metadata_filter_hints(
    question: str,
    *,
    has_version: bool,
    has_dates: bool = False,
) -> tuple[MetadataFilterHint, ...]:
    hints: list[MetadataFilterHint] = []

    if _DOCUMENT_PATTERN.search(question) or _FILENAME_PATTERN.search(question):
        hints.append(MetadataFilterHint.DOCUMENT)
    if has_version:
        hints.append(MetadataFilterHint.DOCUMENT_VERSION)
    if has_dates or _YEAR_PATTERN.search(question) or _DATE_RANGE_PATTERN.search(question):
        hints.append(MetadataFilterHint.DATE_RANGE)
    if _SECTION_PATTERN.search(question):
        hints.append(MetadataFilterHint.SECTION)
    if _FILE_TYPE_PATTERN.search(question):
        hints.append(MetadataFilterHint.FILE_TYPE)
    if _AUTHOR_PATTERN.search(question):
        hints.append(MetadataFilterHint.AUTHOR)

    return tuple(dict.fromkeys(hints))
