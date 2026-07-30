from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from packages.rag_core.ports import LLMProviderError
from packages.rag_core.structured_output import (
    StructuredOutputError,
    StructuredValidationRule,
    generate_structured_output,
)


class _InvalidJsonError(ValueError):
    pass


class _SchemaMismatchError(ValueError):
    pass


class _StubStructuredProvider:
    def __init__(self, *responses: str | BaseException) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    async def generate(self, prompt: str) -> str:
        return prompt

    async def generate_structured(
        self,
        prompt: str,
        *,
        response_schema: Mapping[str, object],
    ) -> str:
        self.calls.append((prompt, response_schema))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _TextOnlyProvider:
    async def generate(self, prompt: str) -> str:
        return prompt


_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"accepted": {"type": "boolean"}},
    "required": ["accepted"],
    "additionalProperties": False,
}
_RULES = (
    StructuredValidationRule(
        failure_code="invalid_json",
        exception_types=(_InvalidJsonError,),
    ),
    StructuredValidationRule(
        failure_code="schema_mismatch",
        exception_types=(_SchemaMismatchError,),
    ),
)


def _parse_accepted(raw_response: str) -> bool:
    if raw_response == '{"accepted":true}':
        return True
    if not raw_response.startswith("{"):
        raise _InvalidJsonError("parser detail must remain private")
    raise _SchemaMismatchError("schema detail must remain private")


@pytest.mark.asyncio
async def test_primary_success_uses_one_call() -> None:
    provider = _StubStructuredProvider('{"accepted":true}')

    result = await generate_structured_output(
        provider=provider,
        prompt="Original task",
        response_schema=_SCHEMA,
        parser=_parse_accepted,
        validation_rules=_RULES,
    )

    assert result.value is True
    assert result.diagnostics.schema_version == "1.0"
    assert result.diagnostics.outcome == "primary_valid"
    assert result.diagnostics.failure_code is None
    assert result.diagnostics.attempt_count == 1
    assert result.diagnostics.repair_attempted is False
    assert provider.calls == [("Original task", _SCHEMA)]


@pytest.mark.asyncio
async def test_declared_validation_failure_repairs_once_with_same_schema() -> None:
    provider = _StubStructuredProvider("malformed", '{"accepted":true}')

    result = await generate_structured_output(
        provider=provider,
        prompt="Original task with permitted evidence",
        response_schema=_SCHEMA,
        parser=_parse_accepted,
        validation_rules=_RULES,
    )

    assert result.value is True
    assert result.diagnostics.outcome == "repair_valid"
    assert result.diagnostics.failure_code == "invalid_json"
    assert result.diagnostics.attempt_count == 2
    assert result.diagnostics.repair_attempted is True
    assert len(provider.calls) == 2
    assert provider.calls[0][1] is _SCHEMA
    assert provider.calls[1][1] is _SCHEMA


@pytest.mark.asyncio
async def test_invalid_primary_and_repair_return_stable_terminal_diagnostics() -> None:
    provider = _StubStructuredProvider("first malformed secret", '{"wrong":"second secret"}')

    with pytest.raises(StructuredOutputError) as captured:
        await generate_structured_output(
            provider=provider,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=_parse_accepted,
            validation_rules=_RULES,
        )

    diagnostics = captured.value.diagnostics
    assert diagnostics.schema_version == "1.0"
    assert diagnostics.outcome == "fallback"
    assert diagnostics.failure_code == "repair_invalid"
    assert diagnostics.attempt_count == 2
    assert diagnostics.repair_attempted is True
    assert len(provider.calls) == 2
    diagnostics_text = repr(diagnostics)
    assert "first malformed secret" not in diagnostics_text
    assert "second secret" not in diagnostics_text
    assert "parser detail must remain private" not in diagnostics_text
    assert "schema detail must remain private" not in diagnostics_text
    assert "first malformed secret" not in str(captured.value)
    assert "second secret" not in str(captured.value)
    assert "parser detail must remain private" not in str(captured.value)
    assert "schema detail must remain private" not in str(captured.value)
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_initial_provider_failure_is_safe_and_does_not_repair() -> None:
    provider = _StubStructuredProvider(LLMProviderError("private provider response"))

    with pytest.raises(StructuredOutputError) as captured:
        await generate_structured_output(
            provider=provider,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=_parse_accepted,
            validation_rules=_RULES,
        )

    assert captured.value.diagnostics.outcome == "fallback"
    assert captured.value.diagnostics.failure_code == "provider_failure"
    assert captured.value.diagnostics.attempt_count == 1
    assert captured.value.diagnostics.repair_attempted is False
    assert len(provider.calls) == 1
    assert "private provider response" not in str(captured.value)
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_repair_provider_failure_is_safe() -> None:
    provider = _StubStructuredProvider(
        "malformed",
        LLMProviderError("private repair provider response"),
    )

    with pytest.raises(StructuredOutputError) as captured:
        await generate_structured_output(
            provider=provider,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=_parse_accepted,
            validation_rules=_RULES,
        )

    assert captured.value.diagnostics.outcome == "fallback"
    assert captured.value.diagnostics.failure_code == "provider_failure"
    assert captured.value.diagnostics.attempt_count == 2
    assert captured.value.diagnostics.repair_attempted is True
    assert len(provider.calls) == 2
    assert "private repair provider response" not in str(captured.value)
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_repair_prompt_omits_raw_response_and_exception_message() -> None:
    malformed = "RAW_MALFORMED_RESPONSE_SECRET"
    provider = _StubStructuredProvider(malformed, '{"accepted":true}')

    await generate_structured_output(
        provider=provider,
        prompt="Original task with permitted evidence",
        response_schema=_SCHEMA,
        parser=_parse_accepted,
        validation_rules=_RULES,
    )

    repair_prompt = provider.calls[1][0]
    assert "Original task with permitted evidence" in repair_prompt
    assert "Validation category: invalid_json." in repair_prompt
    assert "Return only syntactically valid JSON." in repair_prompt
    assert malformed not in repair_prompt
    assert "parser detail must remain private" not in repair_prompt


@pytest.mark.asyncio
async def test_cancellation_propagates_without_repair() -> None:
    provider = _StubStructuredProvider(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await generate_structured_output(
            provider=provider,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=_parse_accepted,
            validation_rules=_RULES,
        )

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_parser_cancellation_propagates_without_repair() -> None:
    provider = _StubStructuredProvider('{"accepted":true}')

    def cancelled_parser(_: str) -> bool:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await generate_structured_output(
            provider=provider,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=cancelled_parser,
            validation_rules=_RULES,
        )

    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_unexpected_provider_and_parser_errors_propagate_without_repair() -> None:
    provider_error = _StubStructuredProvider(RuntimeError("unexpected provider error"))
    with pytest.raises(RuntimeError, match="unexpected provider error"):
        await generate_structured_output(
            provider=provider_error,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=_parse_accepted,
            validation_rules=_RULES,
        )
    assert len(provider_error.calls) == 1

    parser_error = _StubStructuredProvider('{"accepted":true}')

    def unexpected_parser(_: str) -> bool:
        raise RuntimeError("unexpected parser error")

    with pytest.raises(RuntimeError, match="unexpected parser error"):
        await generate_structured_output(
            provider=parser_error,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=unexpected_parser,
            validation_rules=_RULES,
        )
    assert len(parser_error.calls) == 1


@pytest.mark.asyncio
async def test_repair_can_be_disabled_for_rollback() -> None:
    provider = _StubStructuredProvider("malformed")

    with pytest.raises(StructuredOutputError) as captured:
        await generate_structured_output(
            provider=provider,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=_parse_accepted,
            validation_rules=_RULES,
            max_repair_attempts=0,
        )

    assert captured.value.diagnostics.failure_code == "invalid_json"
    assert captured.value.diagnostics.attempt_count == 1
    assert captured.value.diagnostics.repair_attempted is False
    assert len(provider.calls) == 1
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_missing_capability_and_more_than_one_repair_fail_before_generation() -> None:
    with pytest.raises(TypeError, match="structured generation"):
        await generate_structured_output(
            provider=_TextOnlyProvider(),  # type: ignore[arg-type]
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=_parse_accepted,
            validation_rules=_RULES,
        )

    provider = _StubStructuredProvider('{"accepted":true}')
    with pytest.raises(ValueError, match="must be 0 or 1"):
        await generate_structured_output(
            provider=provider,
            prompt="Original task",
            response_schema=_SCHEMA,
            parser=_parse_accepted,
            validation_rules=_RULES,
            max_repair_attempts=2,  # type: ignore[arg-type]
        )
    assert provider.calls == []


def test_structured_output_public_api_is_reexported_from_split_modules() -> None:
    from packages.rag_core import structured_output
    from packages.rag_core.structured_output.models import (
        StructuredOutputDiagnostics as ModelsDiagnostics,
        StructuredOutputError as ModelsError,
        StructuredValidationRule as ModelsValidationRule,
    )
    from packages.rag_core.structured_output.service import (
        generate_structured_output as service_generate_structured_output,
    )

    assert structured_output.StructuredOutputDiagnostics is ModelsDiagnostics
    assert structured_output.StructuredOutputError is ModelsError
    assert structured_output.StructuredValidationRule is ModelsValidationRule
    assert structured_output.generate_structured_output is service_generate_structured_output
