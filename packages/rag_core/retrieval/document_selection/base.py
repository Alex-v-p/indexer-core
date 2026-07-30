from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from packages.rag_core.documents.preferences import DocumentPreference
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.graders import EvidenceGradingReport
from packages.rag_core.retrieval.models import EvidenceItem

if TYPE_CHECKING:
    from packages.rag_core.retrieval.document_selection.models import DocumentCandidateSelection

PRIMARY_DOCUMENT_DETECTOR_TOOL = "document.primary_detector"
DOCUMENT_CANDIDATE_SELECTOR_TOOL = "document.candidate_selector"


class PrimaryDocumentDetector(Protocol):
    """Infer or update a soft primary-document preference from graded evidence."""

    def detect(
        self,
        *,
        question: str,
        information_need: InformationNeed,
        evidence: list[EvidenceItem],
        grading: EvidenceGradingReport,
        existing_preference: DocumentPreference | None = None,
    ) -> DocumentPreference | None:
        """Return the learned preference or preserve the existing one."""


class DocumentCandidateSelector(Protocol):
    """Select a document-balanced final set from a broader candidate pool."""

    @property
    def candidate_multiplier(self) -> int:
        """Candidate expansion used before document balancing."""

    @property
    def max_candidates(self) -> int | None:
        """Upper bound for the expanded candidate pool."""

    def select(
        self,
        evidence: list[EvidenceItem],
        *,
        top_k: int,
        preference: DocumentPreference | None,
    ) -> "DocumentCandidateSelection":
        """Return a final balanced set and trace metadata."""

