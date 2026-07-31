from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BackgroundJobResponse(BaseModel):
    id: uuid.UUID
    job_type: str
    status: str
    priority: int
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    progress: float
    current_stage: str | None = None
    attempts: int
    max_attempts: int
    dedupe_key: str | None = None
    scheduled_at: datetime
    heartbeat_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class EvaluationJobRequest(BaseModel):
    dataset_path: str = Field(min_length=1, examples=["baseline_demo.json"])
    pipeline_name: str | None = None
    top_k: int | None = Field(default=None, gt=0, le=100)
    repetitions: int = Field(default=1, ge=1, le=20)
