from __future__ import annotations

from typing import Protocol

from packages.rag_core.retrieval.retry.models import (
    InformationNeedRetryContext,
    InformationNeedRetryDecision,
    RetrievalRetryContext,
    RetrievalRetryDecision,
)


class RetrievalRetryPolicy(Protocol):
    """Bounded retry controller used by legacy and hierarchical graph flows."""

    name: str
    max_retries: int

    def decide(self, context: RetrievalRetryContext) -> RetrievalRetryDecision:
        """Return the legacy next retry action or deterministic stop decision."""

    def decide_information_need(
        self,
        context: InformationNeedRetryContext,
    ) -> InformationNeedRetryDecision:
        """Route one information need after its latest evidence grade."""
