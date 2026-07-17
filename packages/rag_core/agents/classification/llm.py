from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.agents.classification.base import QueryClassifier
from packages.rag_core.agents.classification.heuristic import HeuristicQueryClassifier
from packages.rag_core.agents.classification.models import MetadataFilterHint, QueryClassification, QueryType
from packages.rag_core.ports import LLMProvider

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "classify_query.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class QueryClassificationError(RuntimeError):
    """Raised when an LLM response cannot be converted into a classification."""


class LLMQueryClassifier:
    """Provider-neutral LLM classifier with a deterministic fail-open path."""

    name = "llm_query_classifier"

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        fallback_classifier: QueryClassifier | None = None,
        fail_open: bool = True,
        max_rationale_chars: int = 500,
    ) -> None:
        if max_rationale_chars <= 0:
            raise ValueError("max_rationale_chars must be positive.")
        self._llm_provider = llm_provider
        self._fallback_classifier = fallback_classifier or HeuristicQueryClassifier()
        self._fail_open = fail_open
        self._max_rationale_chars = max_rationale_chars

    async def classify(self, question: str) -> QueryClassification:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        try:
            raw_response = await self._llm_provider.generate(build_query_classification_prompt(normalized))
            return parse_query_classification(
                raw_response,
                classifier_name=self.name,
                max_rationale_chars=self._max_rationale_chars,
            )
        except Exception as exc:
            if not self._fail_open:
                if isinstance(exc, QueryClassificationError):
                    raise
                raise QueryClassificationError("Query classification failed.") from exc

            fallback = await self._fallback_classifier.classify(normalized)
            return QueryClassification(
                query_type=fallback.query_type,
                confidence=fallback.confidence,
                needs_metadata_filters=fallback.needs_metadata_filters,
                metadata_filter_hints=fallback.metadata_filter_hints,
                rationale=fallback.rationale,
                classifier_name=fallback.classifier_name,
                fallback_used=True,
            )


def build_query_classification_prompt(question: str) -> str:
    """Build the query-classification prompt from the checked-in template."""

    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    return _load_prompt_template().replace("{{ question }}", normalized).strip()


def parse_query_classification(
    raw_response: str,
    *,
    classifier_name: str = LLMQueryClassifier.name,
    max_rationale_chars: int = 500,
) -> QueryClassification:
    """Parse and validate one JSON classification returned by a language model."""

    if max_rationale_chars <= 0:
        raise ValueError("max_rationale_chars must be positive.")
    payload = _extract_json_object(raw_response)

    try:
        query_type = QueryType(str(payload["query_type"]).strip().casefold())
    except (KeyError, ValueError) as exc:
        raise QueryClassificationError("Classification query_type is missing or invalid.") from exc

    confidence = _parse_confidence(payload.get("confidence"))
    hints = _parse_filter_hints(payload.get("metadata_filter_hints", []))
    requested_filters = payload.get("needs_metadata_filters")
    if not isinstance(requested_filters, bool):
        raise QueryClassificationError("needs_metadata_filters must be a boolean.")

    if query_type is QueryType.VERSION_SPECIFIC and MetadataFilterHint.DOCUMENT_VERSION not in hints:
        hints = (*hints, MetadataFilterHint.DOCUMENT_VERSION)
    needs_metadata_filters = requested_filters or bool(hints)

    rationale_value = payload.get("rationale", "")
    rationale = " ".join(str(rationale_value).strip().split())[:max_rationale_chars].strip()
    if not rationale:
        raise QueryClassificationError("Classification rationale must not be empty.")

    return QueryClassification(
        query_type=query_type,
        confidence=confidence,
        needs_metadata_filters=needs_metadata_filters,
        metadata_filter_hints=hints,
        rationale=rationale,
        classifier_name=classifier_name,
    )


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw_response.strip())
    start = cleaned.find("{")
    if start < 0:
        raise QueryClassificationError("Classification response did not contain a JSON object.")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise QueryClassificationError("Classification response contained invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise QueryClassificationError("Classification response must be a JSON object.")
    return parsed


def _parse_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueryClassificationError("confidence must be a number between 0 and 1.")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise QueryClassificationError("confidence must be between 0 and 1.")
    return confidence


def _parse_filter_hints(value: object) -> tuple[MetadataFilterHint, ...]:
    if not isinstance(value, list):
        raise QueryClassificationError("metadata_filter_hints must be a JSON array.")

    hints: list[MetadataFilterHint] = []
    for item in value:
        if not isinstance(item, str):
            raise QueryClassificationError("metadata_filter_hints may only contain strings.")
        try:
            hint = MetadataFilterHint(item.strip().casefold())
        except ValueError as exc:
            allowed = ", ".join(candidate.value for candidate in MetadataFilterHint)
            raise QueryClassificationError(f"Unknown metadata filter hint {item!r}. Allowed: {allowed}.") from exc
        if hint not in hints:
            hints.append(hint)
    return tuple(hints)
