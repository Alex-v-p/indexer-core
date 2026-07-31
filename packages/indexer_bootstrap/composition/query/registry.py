from __future__ import annotations

from packages.indexer_bootstrap.composition.query.agentic import register_agentic_query_pipeline
from packages.indexer_bootstrap.composition.query.pipelines import register_fixed_query_pipelines
from packages.indexer_bootstrap.composition.query.tools import build_query_tool_registry
from packages.indexer_bootstrap.config import Settings
from packages.rag_core.agents.tools import ToolRegistry
from packages.rag_core.pipelines import PipelineRegistry, RetrievalPipeline


def build_query_pipeline_registry(
    settings: Settings,
    *,
    tool_registry: ToolRegistry | None = None,
) -> PipelineRegistry:
    tools = tool_registry or build_query_tool_registry(settings)
    registry = PipelineRegistry(default_pipeline_name=settings.default_query_pipeline)

    register_fixed_query_pipelines(registry, settings=settings, tools=tools)
    register_agentic_query_pipeline(registry, settings=settings, tools=tools)

    registry.validate()
    for pipeline_config in registry.configs():
        for tool_name in pipeline_config.tool_names:
            tools.resolve(tool_name)
    return registry


def build_query_graph(settings: Settings, *, pipeline_name: str | None = None) -> RetrievalPipeline:
    return build_query_pipeline_registry(settings).build(pipeline_name)
