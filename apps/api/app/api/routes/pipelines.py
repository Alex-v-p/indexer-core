from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.pipelines import PipelineListResponse, PipelineSummaryResponse, ToolSummaryResponse
from app.services.query_graph import build_query_pipeline_registry, build_query_tool_registry

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("", response_model=PipelineListResponse)
async def read_pipelines(settings: Settings = Depends(get_settings)) -> PipelineListResponse:
    """List selectable query pipelines and the logical tools they use."""

    tool_registry = build_query_tool_registry(settings)
    pipeline_registry = build_query_pipeline_registry(settings, tool_registry=tool_registry)
    tool_configs = {tool.name: tool for tool in tool_registry.configs()}

    return PipelineListResponse(
        default_pipeline_name=pipeline_registry.default_pipeline_name,
        pipelines=[
            PipelineSummaryResponse(
                name=config.name,
                version=config.version,
                description=config.description,
                is_default=config.name == pipeline_registry.default_pipeline_name,
                tools=[
                    ToolSummaryResponse(
                        name=tool_configs[tool_name].name,
                        kind=tool_configs[tool_name].kind,
                        version=tool_configs[tool_name].version,
                        description=tool_configs[tool_name].description,
                        metadata=tool_configs[tool_name].metadata,
                    )
                    for tool_name in config.tool_names
                ],
                metadata=config.metadata,
            )
            for config in pipeline_registry.configs()
        ],
    )
