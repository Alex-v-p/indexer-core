from __future__ import annotations

from fastapi import Request

from packages.rag_core.agents.tools import ToolRegistry
from packages.rag_core.pipelines import PipelineRegistry


def get_query_tool_registry(request: Request) -> ToolRegistry:
    """Return the application-scoped query tool registry."""

    return request.app.state.query_tool_registry


def get_query_pipeline_registry(request: Request) -> PipelineRegistry:
    """Return the application-scoped query pipeline registry."""

    return request.app.state.query_pipeline_registry
