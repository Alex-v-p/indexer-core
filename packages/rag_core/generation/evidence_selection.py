from __future__ import annotations

from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem


def select_answer_evidence(
    evidence: tuple[EvidenceItem, ...],
    grading: EvidenceGradingReport | None,
) -> tuple[EvidenceItem, ...]:
    """Return only non-empty, grader-approved evidence for generation."""

    if grading is None:
        return tuple(item for item in evidence if item.text.strip())

    relevant_ranks = set(grading.relevant_evidence_ranks)
    return tuple(item for item in evidence if item.rank in relevant_ranks and item.text.strip())
