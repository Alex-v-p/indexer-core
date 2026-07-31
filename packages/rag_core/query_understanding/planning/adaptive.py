from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.ports import StructuredLLMProvider
from packages.rag_core.query_understanding.decomposition.context import (
    contextualize_retrieval_query,
    infer_subject_context,
)
from packages.rag_core.query_understanding.decomposition.lane_distinctness import (
    isolate_retrieval_query_intent,
)
from packages.rag_core.query_understanding.decomposition.models import InformationNeed
from packages.rag_core.query_understanding.planning.base import RetrievalQueryRewriter
from packages.rag_core.query_understanding.planning.models import (
    InformationNeedPlanningContext,
    RetrievalQueryRewrite,
)
from packages.rag_core.structured_output import (
    StructuredOutputError,
    StructuredValidationRule,
    generate_structured_output,
)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "rewrite_retrieval_query.md"


class RetrievalQueryRewriteError(RuntimeError):
    """Raised when adaptive retry-query generation cannot produce valid output."""


class _RewriteInvalidJSONError(RetrievalQueryRewriteError):
    pass


class _RewriteSchemaError(RetrievalQueryRewriteError):
    pass


class _RewriteSemanticError(RetrievalQueryRewriteError):
    pass


_VALIDATION_RULES = (
    StructuredValidationRule(
        failure_code="invalid_json",
        exception_types=(_RewriteInvalidJSONError,),
    ),
    StructuredValidationRule(
        failure_code="schema_mismatch",
        exception_types=(_RewriteSchemaError,),
    ),
    StructuredValidationRule(
        failure_code="semantic_validation_failed",
        exception_types=(_RewriteSemanticError,),
    ),
)


class DeterministicRetrievalQueryRewriter:
    """Safe fail-open retry query builder that preserves the decomposed subject."""

    name = "deterministic_retrieval_query_rewriter"

    def __init__(self, *, max_query_chars: int = 1_200) -> None:
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive.")
        self._max_query_chars = max_query_chars

    async def rewrite(self, context: InformationNeedPlanningContext) -> RetrievalQueryRewrite:
        base = (
            context.previous_queries[-1]
            if context.previous_queries
            else context.information_need.retrieval_query
        )
        subject_context = context.information_need.subject_context or infer_subject_context(
            context.original_question,
            max_chars=min(self._max_query_chars, 240),
        )
        query = contextualize_retrieval_query(
            base,
            subject_context=subject_context,
            max_chars=self._max_query_chars,
        )
        query = _merge_search_terms(
            query,
            context.information_need.description,
            max_chars=self._max_query_chars,
        )
        query = isolate_retrieval_query_intent(
            query,
            information_need=context.information_need,
            sibling_information_needs=context.sibling_information_needs,
            max_query_chars=self._max_query_chars,
        )
        return RetrievalQueryRewrite(
            query=query,
            failure_mode="deterministic_fallback",
            missing_aspects=(context.information_need.description,),
            rationale=(
                "Adaptive generation was unavailable, so the retry preserved the named subject "
                "and added only distinct terms from the information need."
            ),
            rewriter_name=self.name,
            fallback_used=True,
        )


class LLMRetrievalQueryRewriter:
    """Evidence-aware LLM reflection used before each retrieval retry."""

    name = "llm_retrieval_query_rewriter"

    def __init__(
        self,
        *,
        llm_provider: StructuredLLMProvider,
        fallback_rewriter: RetrievalQueryRewriter | None = None,
        fail_open: bool = True,
        max_query_chars: int = 1_200,
        max_rationale_chars: int = 500,
        max_missing_aspects: int = 5,
        max_aspect_chars: int = 180,
        max_attempts_in_prompt: int = 3,
        max_evidence_per_attempt: int = 5,
        max_chars_per_evidence: int = 700,
        max_repair_attempts: int = 1,
    ) -> None:
        if not isinstance(llm_provider, StructuredLLMProvider):
            raise TypeError("llm_provider must support structured generation.")
        if max_query_chars <= 0 or max_rationale_chars <= 0 or max_aspect_chars <= 0:
            raise ValueError("character limits must be positive.")
        if max_missing_aspects <= 0:
            raise ValueError("max_missing_aspects must be positive.")
        if max_attempts_in_prompt <= 0 or max_evidence_per_attempt <= 0:
            raise ValueError("prompt history limits must be positive.")
        if max_chars_per_evidence <= 0:
            raise ValueError("max_chars_per_evidence must be positive.")
        if isinstance(max_repair_attempts, bool) or max_repair_attempts not in (0, 1):
            raise ValueError("max_repair_attempts must be 0 or 1.")
        self._llm_provider = llm_provider
        self._fallback_rewriter = fallback_rewriter or DeterministicRetrievalQueryRewriter(
            max_query_chars=max_query_chars,
        )
        self._fail_open = fail_open
        self._max_query_chars = max_query_chars
        self._max_rationale_chars = max_rationale_chars
        self._max_missing_aspects = max_missing_aspects
        self._max_aspect_chars = max_aspect_chars
        self._max_attempts_in_prompt = max_attempts_in_prompt
        self._max_evidence_per_attempt = max_evidence_per_attempt
        self._max_chars_per_evidence = max_chars_per_evidence
        self._max_repair_attempts = max_repair_attempts

    async def rewrite(self, context: InformationNeedPlanningContext) -> RetrievalQueryRewrite:
        if not context.previous_attempts:
            return await self._fallback_rewriter.rewrite(context)

        subject_context = context.information_need.subject_context or infer_subject_context(
            context.original_question,
            max_chars=min(self._max_query_chars, 240),
        )
        previous_queries = tuple(query.casefold() for query in context.previous_queries)
        try:
            result = await generate_structured_output(
                provider=self._llm_provider,
                prompt=build_retrieval_query_rewrite_prompt(
                    context,
                    max_attempts=self._max_attempts_in_prompt,
                    max_evidence_per_attempt=self._max_evidence_per_attempt,
                    max_chars_per_evidence=self._max_chars_per_evidence,
                ),
                response_schema=retrieval_query_rewrite_response_schema(
                    max_query_chars=self._max_query_chars,
                    max_rationale_chars=self._max_rationale_chars,
                    max_missing_aspects=self._max_missing_aspects,
                    max_aspect_chars=self._max_aspect_chars,
                ),
                parser=lambda raw_response: parse_retrieval_query_rewrite(
                    raw_response,
                    rewriter_name=self.name,
                    subject_context=subject_context,
                    information_need=context.information_need,
                    sibling_information_needs=context.sibling_information_needs,
                    previous_queries=previous_queries,
                    max_query_chars=self._max_query_chars,
                    max_rationale_chars=self._max_rationale_chars,
                    max_missing_aspects=self._max_missing_aspects,
                    max_aspect_chars=self._max_aspect_chars,
                ),
                validation_rules=_VALIDATION_RULES,
                max_repair_attempts=self._max_repair_attempts,
            )
            return replace(result.value, structured_output=result.diagnostics)
        except StructuredOutputError as exc:
            if not self._fail_open:
                raise RetrievalQueryRewriteError(
                    "Retrieval-query rewriting failed structured validation.",
                ) from exc
            fallback = await self._fallback_rewriter.rewrite(context)
            return replace(fallback, structured_output=exc.diagnostics)
        except Exception as exc:
            if not self._fail_open:
                if isinstance(exc, RetrievalQueryRewriteError):
                    raise
                raise RetrievalQueryRewriteError("Retrieval-query rewriting failed.") from exc
            return await self._fallback_rewriter.rewrite(context)


def build_retrieval_query_rewrite_prompt(
    context: InformationNeedPlanningContext,
    *,
    max_attempts: int = 3,
    max_evidence_per_attempt: int = 5,
    max_chars_per_evidence: int = 700,
) -> str:
    if max_attempts <= 0 or max_evidence_per_attempt <= 0 or max_chars_per_evidence <= 0:
        raise ValueError("prompt limits must be positive.")

    attempt_payloads: list[dict[str, Any]] = []
    for attempt in context.previous_attempts[-max_attempts:]:
        evidence_payloads = []
        for evidence in attempt.evidence[:max_evidence_per_attempt]:
            evidence_payloads.append(
                {
                    "document_name": evidence.document_name,
                    "relevant": evidence.relevant,
                    "relevance_score": evidence.relevance_score,
                    "grading_rationale": evidence.grading_rationale,
                    "text": evidence.text[:max_chars_per_evidence],
                },
            )
        attempt_payloads.append(
            {
                "attempt_number": attempt.attempt_number,
                "query": attempt.query,
                "pipeline": attempt.pipeline_name,
                "strategy": attempt.strategy.value,
                "top_k": attempt.top_k,
                "grade_status": attempt.grade_status,
                "coverage_score": attempt.coverage_score,
                "grading_rationale": attempt.grading_rationale,
                "evidence": evidence_payloads,
            },
        )

    payload = {
        "original_question": context.original_question,
        "subject_context": context.information_need.subject_context,
        "information_need": {
            "id": context.information_need.need_id,
            "description": context.information_need.description,
            "initial_retrieval_query": context.information_need.retrieval_query,
        },
        "previous_queries": list(context.previous_queries),
        "excluded_sibling_information_needs": [
            {
                "id": need.need_id,
                "description": need.description,
                "retrieval_query": need.retrieval_query,
            }
            for need in context.sibling_information_needs
        ],
        "previous_attempts": attempt_payloads,
        "hard_scope": {
            "document": context.classification.document_constraint.to_metadata(),
            "version": context.classification.version_constraint.to_metadata(),
            "dates": [constraint.to_metadata() for constraint in context.classification.date_constraints],
        },
        "preferred_document": (
            context.preferred_document.to_metadata()
            if context.preferred_document is not None
            else None
        ),
    }
    return _load_prompt_template().replace(
        "{{ retry_context }}",
        json.dumps(payload, ensure_ascii=False, indent=2),
    ).strip()


def retrieval_query_rewrite_response_schema(
    *,
    max_query_chars: int = 1_200,
    max_rationale_chars: int = 500,
    max_missing_aspects: int = 5,
    max_aspect_chars: int = 180,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rewritten_query": {
                "type": "string",
                "minLength": 1,
                "maxLength": max_query_chars,
            },
            "failure_mode": {
                "type": "string",
                "minLength": 1,
                "maxLength": 80,
            },
            "missing_aspects": {
                "type": "array",
                "maxItems": max_missing_aspects,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_aspect_chars,
                },
            },
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": max_rationale_chars,
            },
        },
        "required": ["rewritten_query", "failure_mode", "missing_aspects", "rationale"],
        "additionalProperties": False,
    }


def parse_retrieval_query_rewrite(
    raw_response: str,
    *,
    rewriter_name: str = LLMRetrievalQueryRewriter.name,
    subject_context: str = "",
    information_need: InformationNeed | None = None,
    sibling_information_needs: tuple[InformationNeed, ...] = (),
    previous_queries: tuple[str, ...] = (),
    max_query_chars: int = 1_200,
    max_rationale_chars: int = 500,
    max_missing_aspects: int = 5,
    max_aspect_chars: int = 180,
) -> RetrievalQueryRewrite:
    try:
        payload = json.loads(raw_response.strip())
    except json.JSONDecodeError as exc:
        raise _RewriteInvalidJSONError("Rewrite response contained invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise _RewriteSchemaError("Rewrite response must be a JSON object.")
    if set(payload) != {"rewritten_query", "failure_mode", "missing_aspects", "rationale"}:
        raise _RewriteSchemaError("Rewrite response fields do not match the schema.")

    query = _required_text(payload.get("rewritten_query"), "rewritten_query", max_query_chars)
    query = contextualize_retrieval_query(
        query,
        subject_context=subject_context,
        max_chars=max_query_chars,
    )
    if information_need is not None:
        query = isolate_retrieval_query_intent(
            query,
            information_need=information_need,
            sibling_information_needs=sibling_information_needs,
            max_query_chars=max_query_chars,
        )
    if query.casefold() in set(previous_queries):
        raise _RewriteSemanticError("rewritten_query must differ from previous queries.")
    failure_mode = _required_text(payload.get("failure_mode"), "failure_mode", 80)
    rationale = _required_text(payload.get("rationale"), "rationale", max_rationale_chars)

    raw_aspects = payload.get("missing_aspects")
    if not isinstance(raw_aspects, list) or len(raw_aspects) > max_missing_aspects:
        raise _RewriteSchemaError("missing_aspects must be a bounded JSON array.")
    aspects = tuple(
        _required_text(value, "missing_aspects item", max_aspect_chars)
        for value in raw_aspects
    )
    if len(aspects) != len(set(aspect.casefold() for aspect in aspects)):
        raise _RewriteSemanticError("missing_aspects must be unique.")

    return RetrievalQueryRewrite(
        query=query,
        failure_mode=failure_mode,
        missing_aspects=aspects,
        rationale=rationale,
        rewriter_name=rewriter_name,
    )


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _required_text(value: object, field_name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise _RewriteSchemaError(f"{field_name} must be a string.")
    if len(value) > max_chars:
        raise _RewriteSchemaError(f"{field_name} exceeds the character limit.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise _RewriteSchemaError(f"{field_name} must not be empty.")
    return normalized


def _merge_search_terms(base: str, requirement: str, *, max_chars: int) -> str:
    tokens = base.split()
    seen = {token.casefold() for token in tokens}
    for token in requirement.split():
        cleaned = token.strip(".,:;!?()[]{}\"'")
        normalized = cleaned.casefold()
        if not cleaned or normalized in seen or normalized in _SEARCH_NOISE:
            continue
        tokens.append(cleaned)
        seen.add(normalized)
    return " ".join(tokens)[:max_chars].rstrip() or base[:max_chars].rstrip()


_SEARCH_NOISE = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "are",
        "describe",
        "determine",
        "do",
        "does",
        "explain",
        "for",
        "identify",
        "in",
        "is",
        "of",
        "on",
        "provide",
        "tell",
        "the",
        "to",
        "what",
        "which",
        "who",
        "why",
    },
)
