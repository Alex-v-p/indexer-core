from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """Minimal async text-generation provider contract."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
