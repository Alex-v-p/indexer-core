from __future__ import annotations

from packages.indexer_application.dto import (
    ChunkIndexRecord,
    CitationRecord,
    ContentGroupAliasRecord,
    ContentGroupRecord,
    DocumentRecord,
    DocumentContentGroupAssignmentRecord,
    DocumentVersionRecord,
    EvidenceRecord,
    QueryRunRecord,
    DocumentSubjectDecisionRecord,
    DocumentTypeDecisionRecord,
    DocumentTypeRecord,
    SubjectAliasRecord,
    SubjectRecord,
    TraceStepRecord,
)
from packages.indexer_application.ports.object_storage import StoredDocumentReference
from packages.indexer_infrastructure.postgres.models import (
    ContentGroup,
    Document,
    DocumentContentGroupAssignment,
    DocumentSubjectDecision,
    DocumentType,
    DocumentTypeDecision,
    QueryRun,
    Subject,
)
from packages.rag_core.subjects import (
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
    SubjectKind,
)
from packages.rag_core.document_organization import (
    ClassificationConfidenceBand,
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentTypeDecisionState,
)


def storage_metadata(stored_file: StoredDocumentReference) -> dict[str, str | int | None]:
    """Map a durable object-store reference to persisted metadata."""

    return {
        "storage_backend": stored_file.storage_backend,
        "original_filename": stored_file.original_filename,
        "content_type": stored_file.content_type,
        "size_bytes": stored_file.size_bytes,
        "checksum_sha256": stored_file.checksum_sha256,
        "bucket_name": stored_file.bucket_name,
        "object_key": stored_file.object_key,
        "storage_uri": stored_file.storage_uri,
    }


def to_document_record(model: Document) -> DocumentRecord:
    """Convert the document aggregate from SQLAlchemy models to an application DTO."""

    return DocumentRecord(
        id=model.id,
        title=model.title,
        original_filename=model.original_filename,
        content_type=model.content_type,
        storage_uri=model.storage_uri,
        size_bytes=model.size_bytes,
        checksum_sha256=model.checksum_sha256,
        status=model.status,
        metadata=dict(model.metadata_ or {}),
        created_at=model.created_at,
        updated_at=model.updated_at,
        versions=tuple(
            DocumentVersionRecord(
                id=item.id,
                version_number=item.version_number,
                storage_uri=item.storage_uri,
                content_type=item.content_type,
                checksum_sha256=item.checksum_sha256,
                parser_name=item.parser_name,
                parser_version=item.parser_version,
                status=item.status,
                metadata=dict(item.metadata_ or {}),
                published_at=item.published_at,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in model.versions
        ),
        chunk_indexes=tuple(
            ChunkIndexRecord(
                id=item.id,
                document_version_id=item.document_version_id,
                ordinal=item.ordinal,
                content_hash=item.content_hash,
                token_count=item.token_count,
                source_page_start=item.source_page_start,
                source_page_end=item.source_page_end,
                section_title=item.section_title,
                qdrant_collection=item.qdrant_collection,
                qdrant_point_id=item.qdrant_point_id,
                metadata=dict(item.metadata_ or {}),
                created_at=item.created_at,
            )
            for item in model.qdrant_chunk_indexes
        ),
    )


def to_query_run_record(model: QueryRun) -> QueryRunRecord:
    """Convert the query-run aggregate from SQLAlchemy models to an application DTO."""

    return QueryRunRecord(
        id=model.id,
        question=model.question,
        answer=model.answer,
        status=model.status,
        pipeline_name=model.pipeline_name,
        pipeline_version=model.pipeline_version,
        top_k=model.top_k,
        started_at=model.started_at,
        completed_at=model.completed_at,
        error_message=model.error_message,
        metadata=dict(model.metadata_ or {}),
        evidence_items=tuple(
            EvidenceRecord(
                id=item.id,
                rank=item.rank,
                score=item.score,
                text=item.text,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                metadata=dict(item.metadata_ or {}),
            )
            for item in model.evidence_items
        ),
        citations=tuple(
            CitationRecord(
                id=item.id,
                citation_index=item.citation_index,
                label=item.label,
                evidence_id=item.evidence_id,
                page_number=item.page_number,
                quote=item.quote,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                metadata=dict(item.metadata_ or {}),
            )
            for item in model.citations
        ),
        trace_steps=tuple(
            TraceStepRecord(
                id=item.id,
                step_order=item.step_order,
                name=item.name,
                step_type=item.step_type,
                status=item.status,
                duration_ms=item.duration_ms,
                input_summary=item.input_summary,
                output_summary=item.output_summary,
                error_message=item.error_message,
                metadata=dict(item.metadata_ or {}),
            )
            for item in model.trace_steps
        ),
    )


def to_subject_record(model: Subject) -> SubjectRecord:
    return SubjectRecord(
        id=model.id,
        kind=SubjectKind(model.kind),
        name=model.name,
        normalized_name=model.normalized_name,
        description=model.description,
        metadata=dict(model.metadata_ or {}),
        created_at=model.created_at,
        updated_at=model.updated_at,
        archived_at=model.archived_at,
        aliases=tuple(
            SubjectAliasRecord(
                id=alias.id,
                subject_id=alias.subject_id,
                name=alias.name,
                normalized_name=alias.normalized_name,
                created_at=alias.created_at,
                archived_at=alias.archived_at,
            )
            for alias in model.aliases
            if alias.archived_at is None
        ),
    )


def to_document_subject_decision_record(
    model: DocumentSubjectDecision,
) -> DocumentSubjectDecisionRecord:
    return DocumentSubjectDecisionRecord(
        id=model.id,
        document_id=model.document_id,
        subject_id=model.subject_id,
        state=DecisionState(model.state),
        control_source=DecisionControlSource(model.control_source),
        confidence=model.confidence,
        confidence_band=(
            ConfidenceBand(model.confidence_band)
            if model.confidence_band is not None
            else None
        ),
        rationale=model.rationale,
        classifier_version=model.classifier_version,
        policy_version=model.policy_version,
        signals=dict(model.signals or {}),
        classified_document_version_id=model.classified_document_version_id,
        revision=model.revision,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def to_document_type_record(model: DocumentType) -> DocumentTypeRecord:
    return DocumentTypeRecord(
        id=model.id,
        key=model.key,
        label=model.label,
        description=model.description,
        metadata=dict(model.metadata_ or {}),
        archived_at=model.archived_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def to_document_type_decision_record(
    model: DocumentTypeDecision,
) -> DocumentTypeDecisionRecord:
    return DocumentTypeDecisionRecord(
        id=model.id,
        document_id=model.document_id,
        document_type_id=model.document_type_id,
        state=DocumentTypeDecisionState(model.state),
        source=ClassificationSource(model.source),
        confidence=model.confidence,
        confidence_band=(
            ClassificationConfidenceBand(model.confidence_band)
            if model.confidence_band is not None
            else None
        ),
        rationale=model.rationale,
        classifier_version=model.classifier_version,
        policy_version=model.policy_version,
        signals=dict(model.signals or {}),
        classified_document_version_id=model.classified_document_version_id,
        revision=model.revision,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def to_content_group_record(model: ContentGroup) -> ContentGroupRecord:
    return ContentGroupRecord(
        id=model.id,
        name=model.name,
        normalized_name=model.normalized_name,
        description=model.description,
        metadata=dict(model.metadata_ or {}),
        archived_at=model.archived_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
        aliases=tuple(
            ContentGroupAliasRecord(
                id=alias.id,
                content_group_id=alias.content_group_id,
                name=alias.name,
                normalized_name=alias.normalized_name,
                archived_at=alias.archived_at,
                created_at=alias.created_at,
            )
            for alias in model.aliases
            if alias.archived_at is None
        ),
    )


def to_document_content_group_assignment_record(
    model: DocumentContentGroupAssignment,
) -> DocumentContentGroupAssignmentRecord:
    return DocumentContentGroupAssignmentRecord(
        document_id=model.document_id,
        content_group_id=model.content_group_id,
        state=ContentGroupAssignmentState(model.state),
        source=ClassificationSource(model.source),
        unresolved_reason=model.unresolved_reason,
        confidence=model.confidence,
        confidence_band=(
            ClassificationConfidenceBand(model.confidence_band)
            if model.confidence_band is not None
            else None
        ),
        rationale=model.rationale,
        classifier_version=model.classifier_version,
        policy_version=model.policy_version,
        signals=dict(model.signals or {}),
        summary_hash=model.summary_hash,
        classified_document_version_id=model.classified_document_version_id,
        revision=model.revision,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
