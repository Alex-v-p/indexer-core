from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from packages.indexer_application.dto.models import QueryRunRecord


MetadataPayload = dict[str, Any]


@dataclass(frozen=True, slots=True)
class QueryExecutionMetadata:
    """Named query metadata that is part of the application-to-presentation contract.

    ``QueryState.metadata`` remains an internal extension bag. This DTO extracts only
    the stable, presentation-relevant entries so callers never need to discover
    arbitrary metadata keys themselves.
    """

    answer_presentation: MetadataPayload | None = None
    classification: MetadataPayload | None = None
    information_need_decomposition: MetadataPayload | None = None
    retrieval_plan: MetadataPayload | None = None
    evidence_grading: MetadataPayload | None = None
    retrieval_retry: MetadataPayload | None = None
    primary_document_preference: MetadataPayload | None = None
    information_need_resolution: MetadataPayload | None = None
    constraint_validation: MetadataPayload | None = None
    evidence_context: MetadataPayload | None = None
    resolved_subject_scope: MetadataPayload | None = None
    product_outcome: str | None = None

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, object]) -> QueryExecutionMetadata:
        return cls(
            answer_presentation=_object_payload(metadata.get("answer_presentation")),
            classification=_object_payload(metadata.get("query_classification")),
            information_need_decomposition=_object_payload(metadata.get("information_need_decomposition")),
            retrieval_plan=_object_payload(metadata.get("retrieval_plan")),
            evidence_grading=_object_payload(metadata.get("evidence_grading")),
            retrieval_retry=_object_payload(metadata.get("retrieval_retry")),
            primary_document_preference=_object_payload(metadata.get("primary_document_preference")),
            information_need_resolution=_object_payload(metadata.get("information_need_resolution")),
            constraint_validation=_object_payload(metadata.get("constraint_validation")),
            evidence_context=_object_payload(metadata.get("evidence_context")),
            resolved_subject_scope=_object_payload(metadata.get("resolved_subject_scope")),
            product_outcome=_optional_string(metadata.get("query_product_outcome")),
        )


@dataclass(frozen=True, slots=True)
class QueryExecutionResult:
    """Persisted query run plus its typed presentation-facing metadata."""

    query_run: QueryRunRecord
    metadata: QueryExecutionMetadata

    @classmethod
    def from_record(cls, query_run: QueryRunRecord) -> QueryExecutionResult:
        return cls(
            query_run=query_run,
            metadata=QueryExecutionMetadata.from_mapping(query_run.metadata),
        )


def _object_payload(value: object) -> MetadataPayload | None:
    if not isinstance(value, dict):
        return None
    return dict(value)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
