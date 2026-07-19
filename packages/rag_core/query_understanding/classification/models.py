from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from packages.rag_core.documents import DocumentNameConstraint, DocumentVersionConstraint
from packages.rag_core.query_understanding.temporal import DocumentDateConstraint


class QueryType(StrEnum):
    """Primary information need expressed by a user query."""

    FACTUAL_LOOKUP = "factual_lookup"
    BROAD_EXPLANATION = "broad_explanation"
    COMPARISON = "comparison"
    VERSION_SPECIFIC = "version_specific"


class MetadataFilterHint(StrEnum):
    """Metadata dimensions that later planning nodes may turn into filters."""

    DOCUMENT = "document"
    DOCUMENT_VERSION = "document_version"
    DATE_RANGE = "date_range"
    SECTION = "section"
    FILE_TYPE = "file_type"
    AUTHOR = "author"


@dataclass(frozen=True, slots=True)
class QueryClassification:
    """Structured query classification stored in QueryState and execution traces."""

    query_type: QueryType
    confidence: float
    needs_metadata_filters: bool
    metadata_filter_hints: tuple[MetadataFilterHint, ...] = ()
    rationale: str = ""
    classifier_name: str = "unknown"
    fallback_used: bool = False
    document_constraint: DocumentNameConstraint = field(default_factory=DocumentNameConstraint)
    version_constraint: DocumentVersionConstraint = field(default_factory=DocumentVersionConstraint)
    date_constraints: tuple[DocumentDateConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if self.metadata_filter_hints and not self.needs_metadata_filters:
            raise ValueError("metadata_filter_hints require needs_metadata_filters=True.")

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for persistence and tracing."""

        return {
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "needs_metadata_filters": self.needs_metadata_filters,
            "metadata_filter_hints": [hint.value for hint in self.metadata_filter_hints],
            "rationale": self.rationale,
            "classifier_name": self.classifier_name,
            "fallback_used": self.fallback_used,
            "document_constraint": self.document_constraint.to_metadata(),
            "version_constraint": self.version_constraint.to_metadata(),
            "date_constraints": [constraint.to_metadata() for constraint in self.date_constraints],
        }
