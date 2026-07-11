from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """Async text-generation capability used by answer-generation nodes."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class LLMProviderError(RuntimeError):
    """Raised when a language-model implementation cannot produce a response."""
