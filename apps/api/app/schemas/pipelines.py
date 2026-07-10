from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolSummaryResponse(BaseModel):
    name: str
    kind: str
    version: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineSummaryResponse(BaseModel):
    name: str
    version: str
    description: str
    is_default: bool
    tools: list[ToolSummaryResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineListResponse(BaseModel):
    default_pipeline_name: str
    pipelines: list[PipelineSummaryResponse] = Field(default_factory=list)
