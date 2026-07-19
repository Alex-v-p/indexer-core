from __future__ import annotations

from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints


class PriorGradeEvidenceArbitrator:
    """Compatibility arbitrator that preserves the already aggregated report."""

    name = "prior_grade_evidence_arbitrator"

    async def arbitrate(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
        prior_grading: EvidenceGradingReport,
        *,
        constraints: RetrievalConstraints | None = None,
    ) -> EvidenceGradingReport:
        del question, evidence, information_needs, constraints
        return prior_grading
