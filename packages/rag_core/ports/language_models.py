from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


class LLMProvider(Protocol):
    """Async text-generation capability used by answer-generation nodes."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


@runtime_checkable
class StructuredLLMProvider(LLMProvider, Protocol):
    """Text-generation provider that can enforce a caller-supplied JSON schema."""

    async def generate_structured(
        self,
        prompt: str,
        *,
        response_schema: Mapping[str, object],
    ) -> str:
        """Generate text constrained by the exact JSON schema."""


class LLMProviderError(RuntimeError):
    """Raised when a language-model implementation cannot produce a response."""
