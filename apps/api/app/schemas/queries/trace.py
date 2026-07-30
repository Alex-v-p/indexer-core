from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class TraceStepResponse(BaseModel):
    id: uuid.UUID | None = None
    step_order: int
    name: str
    step_type: str | None = None
    status: str
    duration_ms: int | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
