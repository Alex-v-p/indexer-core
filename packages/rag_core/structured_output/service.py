from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TypeVar

from packages.rag_core.ports import LLMProviderError, StructuredLLMProvider
from packages.rag_core.structured_output.models import (
    StructuredOutputDiagnostics,
    StructuredOutputError,
    StructuredOutputFailureCode,
    StructuredOutputResult,
    StructuredValidationRule,
)
from packages.rag_core.structured_output.validation import (
    build_repair_prompt,
    validate_rules,
    validation_failure_code,
)

T = TypeVar("T")


async def generate_structured_output(
    *,
    provider: StructuredLLMProvider,
    prompt: str,
    response_schema: Mapping[str, object],
    parser: Callable[[str], T],
    validation_rules: Sequence[StructuredValidationRule],
    max_repair_attempts: Literal[0, 1] = 1,
) -> StructuredOutputResult[T]:
    """Generate, parse, and at most once repair structured control output.

    Only parser exceptions explicitly declared in ``validation_rules`` trigger
    repair. Provider failures become safe terminal errors; cancellation and
    unexpected exceptions propagate unchanged.
    """

    if not isinstance(provider, StructuredLLMProvider):
        raise TypeError("provider must support structured generation.")
    if isinstance(max_repair_attempts, bool) or max_repair_attempts not in (0, 1):
        raise ValueError("max_repair_attempts must be 0 or 1.")

    rules = tuple(validation_rules)
    validate_rules(rules)

    primary_provider_failed = False
    try:
        primary_response = await provider.generate_structured(
            prompt,
            response_schema=response_schema,
        )
    except LLMProviderError:
        primary_provider_failed = True

    if primary_provider_failed:
        raise StructuredOutputError(
            _fallback_diagnostics(
                failure_code="provider_failure",
                attempt_count=1,
                repair_attempted=False,
            ),
        )

    try:
        value = parser(primary_response)
    except LLMProviderError:
        raise
    except Exception as exc:
        validation_code = validation_failure_code(exc, rules)
        if validation_code is None:
            raise
    else:
        return StructuredOutputResult(
            value=value,
            diagnostics=StructuredOutputDiagnostics(
                outcome="primary_valid",
                failure_code=None,
                attempt_count=1,
                repair_attempted=False,
            ),
        )

    if max_repair_attempts == 0:
        raise StructuredOutputError(
            _fallback_diagnostics(
                failure_code=validation_code,
                attempt_count=1,
                repair_attempted=False,
            ),
        ) from None

    repair_prompt = build_repair_prompt(prompt, failure_code=validation_code)
    repair_provider_failed = False
    try:
        repaired_response = await provider.generate_structured(
            repair_prompt,
            response_schema=response_schema,
        )
    except LLMProviderError:
        repair_provider_failed = True

    if repair_provider_failed:
        raise StructuredOutputError(
            _fallback_diagnostics(
                failure_code="provider_failure",
                attempt_count=2,
                repair_attempted=True,
            ),
        )

    repair_validation_failed = False
    try:
        repaired_value = parser(repaired_response)
    except LLMProviderError:
        raise
    except Exception as exc:
        if validation_failure_code(exc, rules) is None:
            raise
        repair_validation_failed = True

    if repair_validation_failed:
        raise StructuredOutputError(
            _fallback_diagnostics(
                failure_code="repair_invalid",
                attempt_count=2,
                repair_attempted=True,
            ),
        )

    return StructuredOutputResult(
        value=repaired_value,
        diagnostics=StructuredOutputDiagnostics(
            outcome="repair_valid",
            failure_code=validation_code,
            attempt_count=2,
            repair_attempted=True,
        ),
    )


def _fallback_diagnostics(
    *,
    failure_code: StructuredOutputFailureCode,
    attempt_count: int,
    repair_attempted: bool,
) -> StructuredOutputDiagnostics:
    return StructuredOutputDiagnostics(
        outcome="fallback",
        failure_code=failure_code,
        attempt_count=attempt_count,
        repair_attempted=repair_attempted,
    )
