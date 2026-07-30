from __future__ import annotations

import json
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.query_understanding.classification.base import QueryClassifier
from packages.rag_core.query_understanding.classification.rules import HeuristicQueryClassifier
from packages.rag_core.query_understanding.classification.models import (
    MetadataFilterHint,
    QueryClassification,
    QueryType,
)
from packages.rag_core.query_understanding.document_naming import detect_document_name_constraint
from packages.rag_core.query_understanding.versioning import detect_document_version_constraint
from packages.rag_core.query_understanding.temporal import detect_document_date_constraints
from packages.rag_core.ports import StructuredLLMProvider
from packages.rag_core.structured_output import (
    StructuredOutputError,
    StructuredValidationRule,
    generate_structured_output,
)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "classify_query.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class QueryClassificationError(RuntimeError):
    """Raised when an LLM response cannot be converted into a classification."""


class _QueryClassificationInvalidJSONError(QueryClassificationError):
    pass


class _QueryClassificationSchemaError(QueryClassificationError):
    pass


_VALIDATION_RULES = (
    StructuredValidationRule(
        failure_code="invalid_json",
        exception_types=(_QueryClassificationInvalidJSONError,),
    ),
    StructuredValidationRule(
        failure_code="schema_mismatch",
        exception_types=(_QueryClassificationSchemaError,),
    ),
)


class LLMQueryClassifier:
    """Provider-neutral LLM classifier with a deterministic fail-open path."""

    name = "llm_query_classifier"

    def __init__(
        self,
        *,
        llm_provider: StructuredLLMProvider,
        fallback_classifier: QueryClassifier | None = None,
        fail_open: bool = True,
        max_rationale_chars: int = 500,
        timezone_name: str = "UTC",
        max_repair_attempts: int = 1,
    ) -> None:
        if max_rationale_chars <= 0:
            raise ValueError("max_rationale_chars must be positive.")
        if not isinstance(llm_provider, StructuredLLMProvider):
            raise TypeError("llm_provider must support structured generation.")
        if isinstance(max_repair_attempts, bool) or max_repair_attempts not in (0, 1):
            raise ValueError("max_repair_attempts must be 0 or 1.")
        self._llm_provider = llm_provider
        self._fallback_classifier = fallback_classifier or HeuristicQueryClassifier(timezone_name=timezone_name)
        self._fail_open = fail_open
        self._max_rationale_chars = max_rationale_chars
        self._timezone_name = timezone_name
        self._max_repair_attempts = max_repair_attempts

    async def classify(self, question: str) -> QueryClassification:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        try:
            result = await generate_structured_output(
                provider=self._llm_provider,
                prompt=build_query_classification_prompt(normalized),
                response_schema=query_classification_response_schema(
                    max_rationale_chars=self._max_rationale_chars,
                ),
                parser=lambda raw_response: parse_query_classification(
                    raw_response,
                    classifier_name=self.name,
                    max_rationale_chars=self._max_rationale_chars,
                ),
                validation_rules=_VALIDATION_RULES,
                max_repair_attempts=self._max_repair_attempts,
            )
            parsed = replace(result.value, structured_output=result.diagnostics)
            return _merge_deterministic_constraints(
                parsed,
                question=normalized,
                timezone_name=self._timezone_name,
            )
        except StructuredOutputError as exc:
            if not self._fail_open:
                raise QueryClassificationError("Query classification failed structured validation.") from exc
            fallback = await self._fallback_classifier.classify(normalized)
            return replace(
                fallback,
                fallback_used=True,
                structured_output=exc.diagnostics,
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
                document_constraint=fallback.document_constraint,
                version_constraint=fallback.version_constraint,
                date_constraints=fallback.date_constraints,
            )


def _merge_deterministic_constraints(
    classification: QueryClassification,
    *,
    question: str,
    timezone_name: str = "UTC",
) -> QueryClassification:
    document_constraint = detect_document_name_constraint(question)
    version_constraint = detect_document_version_constraint(question)
    date_constraints = detect_document_date_constraints(question, timezone_name=timezone_name)
    hints = list(classification.metadata_filter_hints)
    if document_constraint.active and MetadataFilterHint.DOCUMENT not in hints:
        hints.append(MetadataFilterHint.DOCUMENT)
    if version_constraint.active and MetadataFilterHint.DOCUMENT_VERSION not in hints:
        hints.append(MetadataFilterHint.DOCUMENT_VERSION)
    if date_constraints and MetadataFilterHint.DATE_RANGE not in hints:
        hints.append(MetadataFilterHint.DATE_RANGE)

    query_type = classification.query_type
    if version_constraint.active and query_type is not QueryType.COMPARISON:
        query_type = QueryType.VERSION_SPECIFIC

    return replace(
        classification,
        query_type=query_type,
        needs_metadata_filters=classification.needs_metadata_filters or bool(hints),
        metadata_filter_hints=tuple(hints),
        document_constraint=document_constraint,
        version_constraint=version_constraint,
        date_constraints=date_constraints,
    )


def build_query_classification_prompt(question: str) -> str:
    """Build the query-classification prompt from the checked-in template."""

    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    return _load_prompt_template().replace("{{ question }}", normalized).strip()


def query_classification_response_schema(*, max_rationale_chars: int = 500) -> dict[str, Any]:
    if max_rationale_chars <= 0:
        raise ValueError("max_rationale_chars must be positive.")
    return {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": [query_type.value for query_type in QueryType],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "needs_metadata_filters": {"type": "boolean"},
            "metadata_filter_hints": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": [hint.value for hint in MetadataFilterHint],
                },
            },
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": max_rationale_chars,
            },
        },
        "required": [
            "query_type",
            "confidence",
            "needs_metadata_filters",
            "metadata_filter_hints",
            "rationale",
        ],
        "additionalProperties": False,
    }


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
    if set(payload) != {
        "query_type",
        "confidence",
        "needs_metadata_filters",
        "metadata_filter_hints",
        "rationale",
    }:
        raise _QueryClassificationSchemaError("Classification response fields do not match the schema.")

    raw_query_type = payload.get("query_type")
    if not isinstance(raw_query_type, str):
        raise _QueryClassificationSchemaError("Classification query_type must be a string.")
    try:
        query_type = QueryType(raw_query_type)
    except ValueError as exc:
        raise _QueryClassificationSchemaError("Classification query_type is missing or invalid.") from exc

    confidence = _parse_confidence(payload.get("confidence"))
    hints = _parse_filter_hints(payload.get("metadata_filter_hints", []))
    requested_filters = payload.get("needs_metadata_filters")
    if not isinstance(requested_filters, bool):
        raise _QueryClassificationSchemaError("needs_metadata_filters must be a boolean.")

    if query_type is QueryType.VERSION_SPECIFIC and MetadataFilterHint.DOCUMENT_VERSION not in hints:
        hints = (*hints, MetadataFilterHint.DOCUMENT_VERSION)
    needs_metadata_filters = requested_filters or bool(hints)

    rationale_value = payload.get("rationale")
    if not isinstance(rationale_value, str):
        raise _QueryClassificationSchemaError("Classification rationale must be a string.")
    if len(rationale_value) > max_rationale_chars:
        raise _QueryClassificationSchemaError("Classification rationale exceeds the character limit.")
    rationale = " ".join(rationale_value.strip().split())
    if not rationale:
        raise _QueryClassificationSchemaError("Classification rationale must not be empty.")

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
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise _QueryClassificationInvalidJSONError("Classification response contained invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise _QueryClassificationSchemaError("Classification response must be a JSON object.")
    return parsed


def _parse_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _QueryClassificationSchemaError("confidence must be a number between 0 and 1.")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise _QueryClassificationSchemaError("confidence must be between 0 and 1.")
    return confidence


def _parse_filter_hints(value: object) -> tuple[MetadataFilterHint, ...]:
    if not isinstance(value, list):
        raise _QueryClassificationSchemaError("metadata_filter_hints must be a JSON array.")

    hints: list[MetadataFilterHint] = []
    for item in value:
        if not isinstance(item, str):
            raise _QueryClassificationSchemaError("metadata_filter_hints may only contain strings.")
        try:
            hint = MetadataFilterHint(item)
        except ValueError as exc:
            allowed = ", ".join(candidate.value for candidate in MetadataFilterHint)
            raise _QueryClassificationSchemaError(
                f"Unknown metadata filter hint {item!r}. Allowed: {allowed}.",
            ) from exc
        if hint in hints:
            raise _QueryClassificationSchemaError("metadata_filter_hints must not contain duplicates.")
        hints.append(hint)
    return tuple(hints)
