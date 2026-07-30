from __future__ import annotations

from packages.rag_core.structured_output.models import (
    StructuredValidationFailureCode,
    StructuredValidationRule,
)

_REPAIR_INSTRUCTIONS: dict[StructuredValidationFailureCode, str] = {
    "invalid_json": "Return only syntactically valid JSON.",
    "schema_mismatch": "Return JSON that exactly matches the supplied response schema.",
    "semantic_validation_failed": (
        "Return schema-valid JSON whose values satisfy the constraints in the original task."
    ),
}


def validate_rules(rules: tuple[StructuredValidationRule, ...]) -> None:
    declared_types: set[type[Exception]] = set()
    for rule in rules:
        duplicates = declared_types.intersection(rule.exception_types)
        if duplicates:
            raise ValueError("Structured validation exception types may only be declared once.")
        declared_types.update(rule.exception_types)


def validation_failure_code(
    error: Exception,
    rules: tuple[StructuredValidationRule, ...],
) -> StructuredValidationFailureCode | None:
    for rule in rules:
        if isinstance(error, rule.exception_types):
            return rule.failure_code
    return None


def build_repair_prompt(
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
