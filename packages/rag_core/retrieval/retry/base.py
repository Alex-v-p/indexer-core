from __future__ import annotations

from typing import Protocol

from packages.rag_core.retrieval.retry.models import RetrievalRetryContext, RetrievalRetryDecision


class RetrievalRetryPolicy(Protocol):
    """Choose a controlled fallback after evidence grading."""

    name: str
    max_retries: int

    def decide(self, context: RetrievalRetryContext) -> RetrievalRetryDecision:
        """Return the next retry action or a deterministic stop decision."""
