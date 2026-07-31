from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentVersionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class QueryRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TraceStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class DocumentVersionIdentity:
    id: uuid.UUID
    version_number: int
    uploaded_at: datetime | None = None
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.version_number <= 0:
            raise ValueError("version_number must be positive.")


@dataclass(frozen=True, slots=True)
class DocumentVersionRecord:
    id: uuid.UUID
    version_number: int
    storage_uri: str | None
    content_type: str | None
    checksum_sha256: str | None
    parser_name: str | None
    parser_version: str | None
    status: DocumentVersionStatus
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ChunkIndexRecord:
    id: uuid.UUID
    document_version_id: uuid.UUID
    ordinal: int
    content_hash: str | None
    token_count: int | None
    source_page_start: int | None
    source_page_end: int | None
    section_title: str | None
    qdrant_collection: str
    qdrant_point_id: str
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: uuid.UUID
    title: str
    original_filename: str | None
    content_type: str | None
    storage_uri: str | None
    size_bytes: int | None
    checksum_sha256: str | None
    status: DocumentStatus
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    versions: tuple[DocumentVersionRecord, ...] = ()
    chunk_indexes: tuple[ChunkIndexRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    id: uuid.UUID
    rank: int
    score: float | None
    text: str
    qdrant_chunk_index_id: uuid.UUID | None
    document_id: uuid.UUID | None
    document_version_id: uuid.UUID | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CitationRecord:
    id: uuid.UUID
    citation_index: int
    label: str | None
    evidence_id: uuid.UUID | None
    page_number: int | None
    quote: str | None
    qdrant_chunk_index_id: uuid.UUID | None
    document_id: uuid.UUID | None
    document_version_id: uuid.UUID | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TraceStepRecord:
    id: uuid.UUID
    step_order: int
    name: str
    step_type: str | None
    status: TraceStepStatus
    duration_ms: int | None
    input_summary: str | None
    output_summary: str | None
    error_message: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class QueryRunRecord:
    id: uuid.UUID
    question: str
    answer: str | None
    status: QueryRunStatus
    pipeline_name: str | None
    pipeline_version: str | None
    top_k: int | None
    started_at: datetime
    completed_at: datetime | None
    error_message: str | None
    metadata: dict[str, Any]
    evidence_items: tuple[EvidenceRecord, ...] = ()
    citations: tuple[CitationRecord, ...] = ()
    trace_steps: tuple[TraceStepRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkIndexCreate:
    id: uuid.UUID
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    ordinal: int
    content_hash: str | None
    token_count: int | None
    source_page_start: int | None
    source_page_end: int | None
    section_title: str | None
    qdrant_collection: str
    qdrant_point_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
