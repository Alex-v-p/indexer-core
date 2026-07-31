from __future__ import annotations

import json
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.ports import StructuredLLMProvider
from packages.rag_core.query_understanding.decomposition.base import InformationNeedDecomposer
from packages.rag_core.query_understanding.decomposition.context import (
    contextualize_retrieval_query,
    infer_subject_context,
)
from packages.rag_core.query_understanding.decomposition.heuristic import HeuristicInformationNeedDecomposer
from packages.rag_core.query_understanding.decomposition.models import (
    InformationNeed,
    InformationNeedDecomposition,
)
from packages.rag_core.structured_output import (
    StructuredOutputError,
    StructuredValidationRule,
    generate_structured_output,
)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "decompose_information_needs.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class InformationNeedDecompositionError(RuntimeError):
    """Raised when a question cannot be decomposed into structured needs."""


class _InformationNeedInvalidJSONError(InformationNeedDecompositionError):
    pass


class _InformationNeedSchemaError(InformationNeedDecompositionError):
    pass


class _InformationNeedSemanticError(InformationNeedDecompositionError):
    pass


_VALIDATION_RULES = (
    StructuredValidationRule(
        failure_code="invalid_json",
        exception_types=(_InformationNeedInvalidJSONError,),
    ),
    StructuredValidationRule(
        failure_code="schema_mismatch",
        exception_types=(_InformationNeedSchemaError,),
    ),
    StructuredValidationRule(
        failure_code="semantic_validation_failed",
        exception_types=(_InformationNeedSemanticError,),
    ),
)


class LLMInformationNeedDecomposer:
    """LLM-backed requirement decomposition with deterministic fail-open behavior."""

    name = "llm_information_need_decomposer"

    def __init__(
        self,
        *,
        llm_provider: StructuredLLMProvider,
        fallback_decomposer: InformationNeedDecomposer | None = None,
        fail_open: bool = True,
        max_information_needs: int = 6,
        max_need_chars: int = 240,
        max_rationale_chars: int = 500,
        max_repair_attempts: int = 1,
    ) -> None:
        if max_information_needs <= 0:
            raise ValueError("max_information_needs must be positive.")
        if max_need_chars <= 0:
            raise ValueError("max_need_chars must be positive.")
        if max_rationale_chars <= 0:
            raise ValueError("max_rationale_chars must be positive.")
        if not isinstance(llm_provider, StructuredLLMProvider):
            raise TypeError("llm_provider must support structured generation.")
        if isinstance(max_repair_attempts, bool) or max_repair_attempts not in (0, 1):
            raise ValueError("max_repair_attempts must be 0 or 1.")
        self._llm_provider = llm_provider
        self._fallback_decomposer = fallback_decomposer or HeuristicInformationNeedDecomposer(
            max_information_needs=max_information_needs,
        )
        self._fail_open = fail_open
        self._max_information_needs = max_information_needs
        self._max_need_chars = max_need_chars
        self._max_rationale_chars = max_rationale_chars
        self._max_repair_attempts = max_repair_attempts

    async def decompose(self, question: str) -> InformationNeedDecomposition:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        try:
            result = await generate_structured_output(
                provider=self._llm_provider,
                prompt=build_information_need_prompt(
                    normalized,
                    max_information_needs=self._max_information_needs,
                ),
                response_schema=information_need_decomposition_response_schema(
                    max_information_needs=self._max_information_needs,
                    max_need_chars=self._max_need_chars,
                    max_rationale_chars=self._max_rationale_chars,
                ),
                parser=lambda raw_response: parse_information_need_decomposition(
                    raw_response,
                    decomposer_name=self.name,
                    max_information_needs=self._max_information_needs,
                    max_need_chars=self._max_need_chars,
                    max_rationale_chars=self._max_rationale_chars,
                    original_question=normalized,
                ),
                validation_rules=_VALIDATION_RULES,
                max_repair_attempts=self._max_repair_attempts,
            )
            return replace(result.value, structured_output=result.diagnostics)
        except StructuredOutputError as exc:
            if not self._fail_open:
                raise InformationNeedDecompositionError(
                    "Information-need decomposition failed structured validation.",
                ) from exc
            fallback = await self._fallback_decomposer.decompose(normalized)
            return replace(
                fallback,
                fallback_used=True,
                structured_output=exc.diagnostics,
            )
        except Exception as exc:
            if not self._fail_open:
                if isinstance(exc, InformationNeedDecompositionError):
                    raise
                raise InformationNeedDecompositionError("Information-need decomposition failed.") from exc

            fallback = await self._fallback_decomposer.decompose(normalized)
            return InformationNeedDecomposition(
                information_needs=fallback.information_needs,
                rationale=fallback.rationale,
                decomposer_name=fallback.decomposer_name,
                fallback_used=True,
            )


def build_information_need_prompt(
    question: str,
    *,
    max_information_needs: int = 6,
) -> str:
    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    if max_information_needs <= 0:
        raise ValueError("max_information_needs must be positive.")

    return (
        _load_prompt_template()
        .replace("{{ question }}", normalized)
        .replace("{{ max_information_needs }}", str(max_information_needs))
        .strip()
    )


def information_need_decomposition_response_schema(
    *,
    max_information_needs: int = 6,
    max_need_chars: int = 240,
    max_rationale_chars: int = 500,
) -> dict[str, Any]:
    if max_information_needs <= 0:
        raise ValueError("max_information_needs must be positive.")
    if max_need_chars <= 0 or max_rationale_chars <= 0:
        raise ValueError("character limits must be positive.")
    return {
        "type": "object",
        "properties": {
            "information_needs": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_information_needs,
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": max_need_chars,
                        },
                        "retrieval_query": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": max_need_chars,
                        },
                        "subject_context": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": max_need_chars,
                        },
                    },
                    "required": ["description", "retrieval_query", "subject_context"],
                    "additionalProperties": False,
                },
            },
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": max_rationale_chars,
            },
        },
        "required": ["information_needs", "rationale"],
        "additionalProperties": False,
    }


def parse_information_need_decomposition(
    raw_response: str,
    *,
    decomposer_name: str = LLMInformationNeedDecomposer.name,
    max_information_needs: int = 6,
    max_need_chars: int = 240,
    max_rationale_chars: int = 500,
    original_question: str | None = None,
) -> InformationNeedDecomposition:
    if max_information_needs <= 0:
        raise ValueError("max_information_needs must be positive.")
    if max_need_chars <= 0 or max_rationale_chars <= 0:
        raise ValueError("character limits must be positive.")

    payload = _extract_json_object(raw_response)
    if set(payload) != {"information_needs", "rationale"}:
        raise _InformationNeedSchemaError("Decomposition response fields do not match the schema.")
    raw_needs = payload.get("information_needs")
    if not isinstance(raw_needs, list) or not raw_needs:
        raise _InformationNeedSchemaError("information_needs must be a non-empty JSON array.")
    if len(raw_needs) > max_information_needs:
        raise _InformationNeedSchemaError(
            f"information_needs may contain at most {max_information_needs} entries.",
        )

    needs: list[InformationNeed] = []
    seen_queries: set[str] = set()
    for index, raw_need in enumerate(raw_needs, start=1):
        if not isinstance(raw_need, dict):
            raise _InformationNeedSchemaError("Each information need must be a JSON object.")
        raw_need_fields = set(raw_need)
        if raw_need_fields not in (
            {"description", "retrieval_query"},
            {"description", "retrieval_query", "subject_context"},
        ):
            raise _InformationNeedSchemaError("Information-need fields do not match the schema.")
        description = _parse_required_text(
            raw_need.get("description"),
            field_name="description",
            max_chars=max_need_chars,
        )
        retrieval_query = _parse_required_text(
            raw_need.get("retrieval_query"),
            field_name="retrieval_query",
            max_chars=max_need_chars,
        )
        if "subject_context" in raw_need:
            subject_context = _parse_required_text(
                raw_need.get("subject_context"),
                field_name="subject_context",
                max_chars=max_need_chars,
            )
        else:
            subject_context = ""
        if original_question is not None:
            inferred_context = infer_subject_context(original_question, max_chars=max_need_chars)
            if inferred_context:
                subject_context = contextualize_retrieval_query(
                    subject_context or inferred_context,
                    subject_context=inferred_context,
                    max_chars=max_need_chars,
                )
        retrieval_query = contextualize_retrieval_query(
            retrieval_query,
            subject_context=subject_context,
            max_chars=max_need_chars,
        )
        query_key = retrieval_query.casefold()
        if query_key in seen_queries:
            raise _InformationNeedSemanticError(
                "information_needs must not contain duplicate retrieval queries.",
            )
        seen_queries.add(query_key)
        needs.append(
            InformationNeed(
                need_id=f"need_{index}",
                description=description,
                retrieval_query=retrieval_query,
                subject_context=subject_context,
            ),
        )

    rationale = _parse_required_text(
        payload.get("rationale"),
        field_name="rationale",
        max_chars=max_rationale_chars,
    )
    return InformationNeedDecomposition(
        information_needs=tuple(needs),
        rationale=rationale,
        decomposer_name=decomposer_name,
    )


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw_response.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise _InformationNeedInvalidJSONError("Decomposition response contained invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise _InformationNeedSchemaError("Decomposition response must be a JSON object.")
    return parsed


def _parse_required_text(value: object, *, field_name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise _InformationNeedSchemaError(f"{field_name} must be a string.")
    if len(value) > max_chars:
        raise _InformationNeedSchemaError(f"{field_name} exceeds the character limit.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise _InformationNeedSchemaError(f"{field_name} must not be empty.")
    return normalized
