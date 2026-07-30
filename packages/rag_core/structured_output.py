from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from packages.rag_core.ports import (
    LLMProviderError,
    StructuredLLMProvider,
)

StructuredValidationFailureCode = Literal[
    "invalid_json",
    "schema_mismatch",
    "semantic_validation_failed",
]
StructuredOutputFailureCode = Literal[
    "provider_failure",
    "invalid_json",
    "schema_mismatch",
    "semantic_validation_failed",
    "repair_invalid",
]
StructuredOutputOutcome = Literal["primary_valid", "repair_valid", "fallback"]

_VALIDATION_FAILURE_CODES = frozenset(
    {
        "invalid_json",
        "schema_mismatch",
        "semantic_validation_failed",
    },
)
_REPAIR_INSTRUCTIONS: dict[StructuredValidationFailureCode, str] = {
    "invalid_json": "Return only syntactically valid JSON.",
    "schema_mismatch": "Return JSON that exactly matches the supplied response schema.",
    "semantic_validation_failed": (
        "Return schema-valid JSON whose values satisfy the constraints in the original task."
    ),
}

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StructuredValidationRule:
    """Map caller-owned parser exceptions to one safe validation category."""

    failure_code: StructuredValidationFailureCode
    exception_types: tuple[type[Exception], ...]

    def __post_init__(self) -> None:
        if self.failure_code not in _VALIDATION_FAILURE_CODES:
            raise ValueError(f"Unsupported structured validation failure code: {self.failure_code!r}.")
        if not self.exception_types:
            raise ValueError("exception_types must not be empty.")
        for exception_type in self.exception_types:
            if not isinstance(exception_type, type) or not issubclass(exception_type, Exception):
                raise TypeError("Structured validation exception types must derive from Exception.")
            if issubclass(exception_type, LLMProviderError):
                raise ValueError("LLMProviderError cannot be declared as a structured validation error.")


@dataclass(frozen=True, slots=True)
class StructuredOutputDiagnostics:
    """Versioned, response-safe diagnostics for one structured generation cycle."""

    outcome: StructuredOutputOutcome
    failure_code: StructuredOutputFailureCode | None
    attempt_count: int
    repair_attempted: bool
    schema_version: Literal["1.0"] = "1.0"

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "failure_code": self.failure_code,
            "attempt_count": self.attempt_count,
            "repair_attempted": self.repair_attempted,
        }


@dataclass(frozen=True, slots=True)
class StructuredOutputResult(Generic[T]):
    """Parsed structured value plus safe generation diagnostics."""

    value: T
    diagnostics: StructuredOutputDiagnostics


class StructuredOutputError(RuntimeError):
    """Terminal handled failure that intentionally omits model and parser output."""

    def __init__(self, diagnostics: StructuredOutputDiagnostics) -> None:
        super().__init__(
            f"Structured output generation ended with failure code "
            f"{diagnostics.failure_code or 'unknown'}.",
        )
        self.diagnostics = diagnostics


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
    _validate_rules(rules)

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
        validation_code = _validation_failure_code(exc, rules)
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

    repair_prompt = _build_repair_prompt(prompt, failure_code=validation_code)
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
        if _validation_failure_code(exc, rules) is None:
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


def _validate_rules(rules: tuple[StructuredValidationRule, ...]) -> None:
    declared_types: set[type[Exception]] = set()
    for rule in rules:
        duplicates = declared_types.intersection(rule.exception_types)
        if duplicates:
            raise ValueError("Structured validation exception types may only be declared once.")
        declared_types.update(rule.exception_types)


def _validation_failure_code(
    error: Exception,
    rules: tuple[StructuredValidationRule, ...],
) -> StructuredValidationFailureCode | None:
    for rule in rules:
        if isinstance(error, rule.exception_types):
            return rule.failure_code
    return None


def _build_repair_prompt(
    original_prompt: str,
    *,
    failure_code: StructuredValidationFailureCode,
) -> str:
    return (
        f"{original_prompt}\n\n"
        "The previous response failed structured validation.\n"
        f"Validation category: {failure_code}.\n"
        f"Repair instruction: {_REPAIR_INSTRUCTIONS[failure_code]}\n"
        "Produce one corrected response without commentary."
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


__all__ = [
    "StructuredOutputDiagnostics",
    "StructuredOutputError",
    "StructuredOutputFailureCode",
    "StructuredOutputOutcome",
    "StructuredOutputResult",
    "StructuredValidationFailureCode",
    "StructuredValidationRule",
    "generate_structured_output",
]
