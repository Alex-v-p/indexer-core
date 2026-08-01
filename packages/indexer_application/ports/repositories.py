from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Protocol

from packages.indexer_application.dto import (
    ChunkIndexCreate,
    DocumentRecord,
    DocumentSubjectDecisionRecord,
    DocumentVersionDeletionOutcome,
    DocumentVersionIdentity,
    QueryRunRecord,
    SubjectAliasRecord,
    SubjectNameMatchRecord,
    SubjectRecord,
)
from packages.indexer_application.ports.object_storage import StoredDocumentReference
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.runtime import TraceEvent
from packages.rag_core.subjects import (
    DecisionState,
    DocumentSubjectDecision,
    SubjectKind,
)


class DocumentRepository(Protocol):
    async def create_processing_document(self, *, stored_file: StoredDocumentReference, title: str) -> uuid.UUID: ...

    async def create_processing_version(
        self,
        *,
        document_id: uuid.UUID,
        stored_file: StoredDocumentReference,
        published_at: datetime | None = None,
    ) -> DocumentVersionIdentity: ...

    async def find_version_candidate(
        self,
        *,
        title: str,
        original_filename: str,
    ) -> DocumentRecord | None: ...

    async def set_version_parser_metadata(
        self,
        *,
        version_id: uuid.UUID,
        parser_name: str,
        parser_version: str,
        metadata: dict[str, Any],
    ) -> None: ...

    async def add_chunk_indexes(self, chunks: list[ChunkIndexCreate]) -> None: ...

    async def delete_chunk_indexes(self, *, version_id: uuid.UUID) -> tuple[str, ...]: ...

    async def mark_ready(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        document_metadata: dict[str, Any],
        stored_file: StoredDocumentReference,
        version_number: int,
    ) -> None: ...

    async def mark_failed(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        error_message: str,
    ) -> None: ...

    async def set_subject_classification_status(
        self,
        *,
        document_id: uuid.UUID,
        status: dict[str, Any],
        expected_job_id: uuid.UUID | None = None,
        expected_document_version_id: uuid.UUID | None = None,
        expected_policy_version: str | None = None,
        expected_statuses: tuple[str, ...] | None = None,
    ) -> bool: ...

    async def get(self, document_id: uuid.UUID) -> DocumentRecord | None: ...

    async def get_for_update(self, document_id: uuid.UUID) -> DocumentRecord | None: ...

    async def list(self, *, limit: int, offset: int) -> list[DocumentRecord]: ...

    async def delete_version(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> DocumentVersionDeletionOutcome | None: ...


class SubjectDecisionConflictError(RuntimeError):
    """A decision write lost an optimistic race or violated manual control."""


class SubjectCanonicalNameConflictError(RuntimeError):
    """A subject kind already owns the normalized canonical name."""


class SubjectRepository(Protocol):
    async def create(
        self,
        *,
        kind: SubjectKind,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubjectRecord: ...

    async def get(
        self,
        subject_id: uuid.UUID,
        *,
        include_archived: bool = False,
    ) -> SubjectRecord | None: ...

    async def list(
        self,
        *,
        kind: SubjectKind | None = None,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SubjectRecord]: ...

    async def rename(self, *, subject_id: uuid.UUID, name: str) -> SubjectRecord: ...

    async def archive(self, *, subject_id: uuid.UUID) -> SubjectRecord: ...

    async def add_alias(
        self,
        *,
        subject_id: uuid.UUID,
        name: str,
    ) -> SubjectAliasRecord: ...

    async def archive_alias(
        self,
        *,
        subject_id: uuid.UUID,
        alias_id: uuid.UUID,
    ) -> SubjectAliasRecord: ...

    async def resolve_name(
        self,
        name: str,
        *,
        kind: SubjectKind | None = None,
        include_archived: bool = False,
    ) -> list[SubjectNameMatchRecord]: ...

    async def get_decision(
        self,
        *,
        document_id: uuid.UUID,
        subject_id: uuid.UUID,
    ) -> DocumentSubjectDecisionRecord | None: ...

    async def list_document_decisions(
        self,
        *,
        document_id: uuid.UUID,
        states: tuple[DecisionState, ...] | None = None,
    ) -> list[DocumentSubjectDecisionRecord]: ...

    async def list_document_suggestions(
        self,
        *,
        document_id: uuid.UUID,
    ) -> list[DocumentSubjectDecisionRecord]:
        """List active-subject automatic suggestions only."""

        ...

    async def write_decision(
        self,
        decision: DocumentSubjectDecision,
        *,
        expected_revision: int | None = None,
    ) -> DocumentSubjectDecisionRecord:
        """Persist a current decision with explicit manual concurrency control.

        Manual decisions require ``expected_revision``: zero creates only when
        absent and a positive value updates only that revision. Automatic
        classification may omit it for an atomic, idempotent upsert. Any failed
        explicit comparison-and-swap raises ``SubjectDecisionConflictError``.
        """

        ...

    async def list_assigned_document_ids(
        self,
        *,
        subject_ids: tuple[uuid.UUID, ...],
        require_all: bool = False,
    ) -> tuple[uuid.UUID, ...]: ...


class QueryRunRepository(Protocol):
    async def create_pending(
        self,
        *,
        question: str,
        pipeline_name: str,
        pipeline_version: str,
        top_k: int,
        requested_pipeline_name: str | None,
        requested_subject_ids: tuple[uuid.UUID, ...] = (),
        coverage_mode: str = "best_evidence",
    ) -> uuid.UUID: ...

    async def create_running(
        self,
        *,
        question: str,
        pipeline_name: str,
        pipeline_version: str,
        top_k: int,
        requested_pipeline_name: str | None,
    ) -> uuid.UUID: ...

    async def mark_running(
        self,
        *,
        query_run_id: uuid.UUID,
        pipeline_name: str,
        pipeline_version: str,
        background_job_id: uuid.UUID,
        attempt: int,
    ) -> None: ...

    async def mark_retry_pending(
        self,
        *,
        query_run_id: uuid.UUID,
        error_message: str,
        retry_at: datetime,
    ) -> None: ...

    async def mark_failed(
        self,
        *,
        query_run_id: uuid.UUID,
        error_message: str,
        trace: list[TraceEvent],
    ) -> None: ...

    async def mark_succeeded(self, *, query_run_id: uuid.UUID, state: QueryState) -> None: ...

    async def get(self, query_run_id: uuid.UUID) -> QueryRunRecord | None: ...

    async def get_for_update(self, query_run_id: uuid.UUID) -> QueryRunRecord | None: ...

    async def set_subject_scope_snapshot(
        self,
        *,
        query_run_id: uuid.UUID,
        snapshot: dict[str, Any],
    ) -> None: ...
