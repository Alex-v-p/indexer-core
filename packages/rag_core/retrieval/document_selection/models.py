from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.rag_core.retrieval.models import EvidenceItem


@dataclass(frozen=True, slots=True)
class DocumentCandidateSelection:
    """Balanced evidence selection plus trace-friendly decision metadata."""

    evidence: tuple[EvidenceItem, ...]
    candidate_count: int
    requested_top_k: int
    selected_document_counts: tuple[tuple[str, int], ...]
    primary_document_key: str | None
    primary_selected_count: int
    quota_relaxed: bool
    selector_name: str

    def __post_init__(self) -> None:
        if self.candidate_count < 0:
            raise ValueError("candidate_count must not be negative.")
        if self.requested_top_k <= 0:
            raise ValueError("requested_top_k must be positive.")
        if len(self.evidence) > self.requested_top_k:
            raise ValueError("Selection cannot exceed requested_top_k.")
        if self.primary_selected_count < 0:
            raise ValueError("primary_selected_count must not be negative.")
        if not self.selector_name.strip():
            raise ValueError("selector_name must not be empty.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "selector_name": self.selector_name,
            "candidate_count": self.candidate_count,
            "requested_top_k": self.requested_top_k,
            "selected_count": len(self.evidence),
            "selected_document_counts": dict(self.selected_document_counts),
            "primary_document_key": self.primary_document_key,
            "primary_selected_count": self.primary_selected_count,
            "quota_relaxed": self.quota_relaxed,
        }
