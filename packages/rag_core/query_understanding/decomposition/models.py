from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.rag_core.document_scope import CoverageMode, DocumentScope, SubjectDocumentLane
from packages.rag_core.structured_output import StructuredOutputDiagnostics


@dataclass(frozen=True, slots=True)
class InformationNeed:
    """One atomic answer requirement extracted from the submitted question.

    Information needs describe what retrieved evidence must establish before an
    answer is generated. They are independent of the retrieval strategy chosen
    to satisfy them.
    """

    need_id: str
    description: str
    retrieval_query: str
    subject_context: str = ""
    required: bool = True
    document_scope: DocumentScope = DocumentScope()
    subject_lane: SubjectDocumentLane | None = None
    coverage_mode: CoverageMode = CoverageMode.BEST_EVIDENCE

    def __post_init__(self) -> None:
        if not self.need_id.strip():
            raise ValueError("need_id must not be empty.")
        if not self.description.strip():
            raise ValueError("description must not be empty.")
        if not self.retrieval_query.strip():
            raise ValueError("retrieval_query must not be empty.")
        if self.subject_context and not self.subject_context.strip():
            raise ValueError("subject_context must be empty or contain non-whitespace text.")
        if self.subject_lane is not None and self.document_scope != self.subject_lane.document_scope:
            raise ValueError("Information-need scope must match its subject lane.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "need_id": self.need_id,
            "description": self.description,
            "retrieval_query": self.retrieval_query,
            "subject_context": self.subject_context,
            "required": self.required,
            "document_scope": self.document_scope.to_metadata(),
            "subject_lane": self.subject_lane.to_metadata() if self.subject_lane is not None else None,
            "coverage_mode": self.coverage_mode.value,
        }


@dataclass(frozen=True, slots=True)
class InformationNeedDecomposition:
    """Structured query-understanding result produced before retrieval planning."""

    information_needs: tuple[InformationNeed, ...]
    rationale: str
    decomposer_name: str
    fallback_used: bool = False
    structured_output: StructuredOutputDiagnostics | None = None

    def __post_init__(self) -> None:
        if not self.information_needs:
            raise ValueError("At least one information need is required.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.decomposer_name.strip():
            raise ValueError("decomposer_name must not be empty.")
        ids = [need.need_id for need in self.information_needs]
        if len(ids) != len(set(ids)):
            raise ValueError("Information needs must have unique ids.")

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "information_needs": [need.to_metadata() for need in self.information_needs],
            "information_need_count": len(self.information_needs),
            "rationale": self.rationale,
            "decomposer_name": self.decomposer_name,
            "fallback_used": self.fallback_used,
        }
        if self.structured_output is not None:
            metadata["structured_output"] = self.structured_output.to_metadata()
        return metadata
