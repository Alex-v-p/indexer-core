from __future__ import annotations


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot produce a valid response."""
