from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class BackgroundJobType(StrEnum):
    INGEST_DOCUMENT = "ingest_document"
    REBUILD_DOCUMENT_INDEX = "rebuild_document_index"
    CONTEXTUALIZE_DOCUMENT = "contextualize_document"
    RUN_EVALUATION = "run_evaluation"
    DELETE_DOCUMENT = "delete_document"


class BackgroundJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class BackgroundJobRecord:
    id: uuid.UUID
    job_type: BackgroundJobType
    status: BackgroundJobStatus
    priority: int
    payload: dict[str, Any]
    result: dict[str, Any]
    progress: float
    current_stage: str | None
    attempts: int
    max_attempts: int
    dedupe_key: str | None
    scheduled_at: datetime
    locked_at: datetime | None
    locked_by: str | None
    heartbeat_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class BackgroundJobSubmission:
    job_type: BackgroundJobType
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    max_attempts: int = 3
    dedupe_key: str | None = None
    scheduled_at: datetime | None = None
