from __future__ import annotations

from typing import Protocol

from packages.rag_core.query_understanding.planning import InformationNeed
from packages.rag_core.retrieval.graders.models import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem

EVIDENCE_GRADER_TOOL = "grader.evidence_relevance"


class EvidenceGrader(Protocol):
    """Grade retrieved evidence for relevance and answer sufficiency."""

    async def grade(self, question: str, evidence: list[EvidenceItem]) -> EvidenceGradingReport:
        """Return chunk relevance and a compatible aggregate decision."""


class InformationNeedEvidenceGrader(EvidenceGrader, Protocol):
    """Extended grader that evaluates every planned answer requirement."""

    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
    ) -> EvidenceGradingReport:
        """Return per-chunk and per-information-need coverage grades."""
