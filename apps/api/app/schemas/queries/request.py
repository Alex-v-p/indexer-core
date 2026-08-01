from __future__ import annotations

from datetime import datetime
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=25)
    scheduled_at: datetime | None = Field(
        default=None,
        description="Optional timezone-aware time after which a worker may claim the query job.",
    )
    pipeline_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_.-]*$",
        description="Registered pipeline to run. Omit to use the configured default.",
    )
    subject_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)
    coverage_mode: Literal["best_evidence", "multi_document"] = "best_evidence"
