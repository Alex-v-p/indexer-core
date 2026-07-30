from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from packages.rag_core.ports import StructuredLLMProvider
from packages.rag_core.structured_output import (
    StructuredOutputDiagnostics,
    StructuredOutputError,
    StructuredValidationRule,
    generate_structured_output,
)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "generate_query_variants.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class QueryVariantGenerationError(RuntimeError):
    """Raised when query expansion cannot produce usable query variants."""

    def __init__(
        self,
        message: str,
        *,
        structured_output: StructuredOutputDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.structured_output = structured_output


class _QueryVariantInvalidJSONError(QueryVariantGenerationError):
    pass


class _QueryVariantSchemaError(QueryVariantGenerationError):
    pass


class _QueryVariantSemanticError(QueryVariantGenerationError):
    pass


_VALIDATION_RULES = (
    StructuredValidationRule(
        failure_code="invalid_json",
        exception_types=(_QueryVariantInvalidJSONError,),
    ),
    StructuredValidationRule(
        failure_code="schema_mismatch",
        exception_types=(_QueryVariantSchemaError,),
    ),
    StructuredValidationRule(
        failure_code="semantic_validation_failed",
        exception_types=(_QueryVariantSemanticError,),
    ),
)


class QueryVariantGenerator(Protocol):
    """Generate alternative search queries for one user question."""

    async def generate(self, question: str, *, count: int) -> list[str]:
        """Return up to ``count`` distinct alternatives to the original query."""


@runtime_checkable
class MetadataQueryVariantGenerator(QueryVariantGenerator, Protocol):
    """Query variant generator that returns request-local diagnostics."""

    async def generate_with_metadata(
        self,
        question: str,
        *,
        count: int,
    ) -> QueryVariantGenerationResult:
        """Return variants and diagnostics without shared last-call state."""


@dataclass(frozen=True, slots=True)
class QueryVariantGenerationResult:
    variants: tuple[str, ...]
    structured_output: StructuredOutputDiagnostics


class LLMQueryVariantGenerator:
    """Generate search-oriented query variants through structured generation."""

    def __init__(
        self,
        *,
        llm_provider: StructuredLLMProvider,
        max_variant_chars: int = 300,
        max_repair_attempts: int = 1,
    ) -> None:
        if max_variant_chars <= 0:
            raise ValueError("max_variant_chars must be positive.")
        if not isinstance(llm_provider, StructuredLLMProvider):
            raise TypeError("llm_provider must support structured generation.")
        if isinstance(max_repair_attempts, bool) or max_repair_attempts not in (0, 1):
            raise ValueError("max_repair_attempts must be 0 or 1.")
        self._llm_provider = llm_provider
        self._max_variant_chars = max_variant_chars
        self._max_repair_attempts = max_repair_attempts

    async def generate(self, question: str, *, count: int) -> list[str]:
        result = await self.generate_with_metadata(question, count=count)
        return list(result.variants)

    async def generate_with_metadata(
        self,
        question: str,
        *,
        count: int,
    ) -> QueryVariantGenerationResult:
        normalized_question = " ".join(question.strip().split())
        if not normalized_question:
            raise ValueError("question must not be empty.")
        if count <= 0:
            raise ValueError("count must be positive.")

        terminal_error: StructuredOutputError | None = None
        try:
            result = await generate_structured_output(
                provider=self._llm_provider,
                prompt=build_query_variant_prompt(normalized_question, count=count),
                response_schema=query_variant_response_schema(
                    count=count,
                    max_variant_chars=self._max_variant_chars,
                ),
                parser=lambda raw_response: parse_query_variants(
                    raw_response,
                    original_question=normalized_question,
                    count=count,
                    max_variant_chars=self._max_variant_chars,
                ),
                validation_rules=_VALIDATION_RULES,
                max_repair_attempts=self._max_repair_attempts,
            )
        except StructuredOutputError as exc:
            terminal_error = exc

        if terminal_error is not None:
            raise QueryVariantGenerationError(
                "Structured query-variant generation failed.",
                structured_output=terminal_error.diagnostics,
            )

        return QueryVariantGenerationResult(
            variants=tuple(result.value),
            structured_output=result.diagnostics,
        )


def build_query_variant_prompt(question: str, *, count: int) -> str:
    """Build the deterministic query-expansion prompt from the markdown template."""

    if count <= 0:
        raise ValueError("count must be positive.")
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty.")

    return (
        _load_prompt_template()
        .replace("{{variant_count}}", str(count))
        .replace("{{question}}", normalized_question)
    )


def query_variant_response_schema(*, count: int, max_variant_chars: int = 300) -> dict[str, object]:
    if count <= 0:
        raise ValueError("count must be positive.")
    if max_variant_chars <= 0:
        raise ValueError("max_variant_chars must be positive.")
    return {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 1,
                "maxItems": count,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_variant_chars,
                },
            },
        },
        "required": ["queries"],
        "additionalProperties": False,
    }


def parse_query_variants(
    raw_response: str,
    *,
    original_question: str,
    count: int,
    max_variant_chars: int = 300,
) -> list[str]:
    """Parse and semantically validate the exact structured query-variant shape."""

    if count <= 0:
        raise ValueError("count must be positive.")
    if max_variant_chars <= 0:
        raise ValueError("max_variant_chars must be positive.")

    payload = _extract_json_object(raw_response)
    if set(payload) != {"queries"}:
        raise _QueryVariantSchemaError("Query-variant response fields do not match the schema.")
    candidates = payload["queries"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= count:
        raise _QueryVariantSchemaError("queries must be a non-empty bounded JSON array.")

    original_key = _query_key(original_question)
    variants: list[str] = []
    seen = {original_key}
    for candidate in candidates:
        if not isinstance(candidate, str):
            raise _QueryVariantSchemaError("queries may only contain strings.")
        normalized = _normalize_query(candidate)
        if not normalized or len(candidate) > max_variant_chars:
            raise _QueryVariantSchemaError("query variants must be non-empty and within the character limit.")
        key = _query_key(normalized)
        if key in seen:
            raise _QueryVariantSemanticError(
                "query variants must be distinct from one another and the original question.",
            )
        seen.add(key)
        variants.append(normalized)
    return variants


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _extract_json_object(raw_response: str) -> Mapping[str, object]:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw_response.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise _QueryVariantInvalidJSONError("Query-variant response contained invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise _QueryVariantSchemaError("Query-variant response must be a JSON object.")
    return parsed


def _normalize_query(value: str) -> str:
    return " ".join(value.strip().split())


def _query_key(value: str) -> str:
    return " ".join(value.casefold().split())


__all__ = [
    "LLMQueryVariantGenerator",
    "MetadataQueryVariantGenerator",
    "QueryVariantGenerationError",
    "QueryVariantGenerationResult",
    "QueryVariantGenerator",
    "build_query_variant_prompt",
    "parse_query_variants",
    "query_variant_response_schema",
]
