from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentVersionResponse(BaseModel):
    id: uuid.UUID
    version_number: int
    storage_uri: str | None = None
    content_type: str | None = None
    checksum_sha256: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    status: str
    is_latest: bool = False
    uploaded_at: datetime
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ChunkIndexResponse(BaseModel):
    id: uuid.UUID
    ordinal: int
    content_hash: str | None = None
    token_count: int | None = None
    source_page_start: int | None = None
    source_page_end: int | None = None
    section_title: str | None = None
    qdrant_collection: str
    qdrant_point_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DocumentSummaryResponse(BaseModel):
    id: uuid.UUID
    title: str
    original_filename: str | None = None
    content_type: str | None = None
    storage_uri: str | None = None
    size_bytes: int | None = None
    checksum_sha256: str | None = None
    status: str
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(DocumentSummaryResponse):
    versions: list[DocumentVersionResponse] = Field(default_factory=list)
    chunks: list[ChunkIndexResponse] = Field(default_factory=list)


class QueuedDocumentUploadItemResponse(BaseModel):
    filename: str
    document: DocumentDetailResponse
    job_id: uuid.UUID


class RejectedDocumentUploadResponse(BaseModel):
    filename: str
    detail: str


class BatchDocumentUploadResponse(BaseModel):
    accepted: list[QueuedDocumentUploadItemResponse] = Field(default_factory=list)
    rejected: list[RejectedDocumentUploadResponse] = Field(default_factory=list)


class DocumentVersionDeletionTargetRequest(BaseModel):
    document_id: uuid.UUID
    document_version_id: uuid.UUID


class BatchDocumentVersionDeletionRequest(BaseModel):
    targets: list[DocumentVersionDeletionTargetRequest] = Field(min_length=1, max_length=100)


class QueuedDocumentVersionDeletionResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    target_count: int
