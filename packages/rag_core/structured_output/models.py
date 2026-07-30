from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from packages.rag_core.ports import LLMProviderError

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
