from __future__ import annotations

from typing import Protocol

from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints

EVIDENCE_ARBITRATOR_TOOL = "arbiter.final_evidence"


class EvidenceArbitrator(Protocol):
    """Perform the final question-level evidence decision before generation."""

    async def arbitrate(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
        prior_grading: EvidenceGradingReport,
        *,
        constraints: RetrievalConstraints | None = None,
    ) -> EvidenceGradingReport:
        """Return a final report that may only preserve or downgrade prior support."""
