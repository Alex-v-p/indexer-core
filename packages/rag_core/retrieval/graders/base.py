from __future__ import annotations

from typing import Protocol

from packages.rag_core.retrieval.graders.models import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem

EVIDENCE_GRADER_TOOL = "grader.evidence_relevance"


class EvidenceGrader(Protocol):
    """Grade retrieved evidence for relevance and answer sufficiency."""

    async def grade(self, question: str, evidence: list[EvidenceItem]) -> EvidenceGradingReport:
        """Return per-item relevance grades and one aggregate sufficiency decision."""
