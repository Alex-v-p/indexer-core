from __future__ import annotations

import pytest

from app.core.config import Settings
from app.composition import build_query_graph, build_query_pipeline_registry, build_query_tool_registry
from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL
from packages.rag_core.query_understanding.decomposition import INFORMATION_NEED_DECOMPOSER_TOOL
from packages.rag_core.query_understanding.planning import RETRIEVAL_PLANNER_TOOL
from packages.rag_core.retrieval.graders import EVIDENCE_GRADER_TOOL
from packages.rag_core.agents.state import QueryState
from packages.rag_core.agents.tools import DuplicateToolError, ToolConfig, ToolRegistry, UnknownToolError
from packages.rag_core.pipelines import (
    AGENTIC_RAG_NAME,
    AGENTIC_RAG_VERSION,
    BASELINE_LLM_TOOL,
    BASELINE_RAG_NAME,
    BASELINE_RAG_VERSION,
    BASELINE_RETRIEVER_TOOL,
    CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
    CONTEXTUAL_RAG_NAME,
    CONTEXTUAL_RAG_VERSION,
    CONTEXTUAL_RETRIEVER_TOOL,
    CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
    DuplicatePipelineError,
    HYBRID_CROSS_ENCODER_RERANKER_TOOL,
    HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    HYBRID_CROSS_ENCODER_RERANK_RAG_VERSION,
    HYBRID_KEYWORD_RETRIEVER_TOOL,
    HYBRID_RAG_NAME,
    HYBRID_RAG_VERSION,
    HYBRID_LLM_RERANKER_TOOL,
    HYBRID_LLM_RERANK_RAG_NAME,
    HYBRID_LLM_RERANK_RAG_VERSION,
    HYBRID_RETRIEVER_TOOL,
    MULTI_QUERY_GENERATOR_TOOL,
    MULTI_QUERY_RAG_NAME,
    MULTI_QUERY_RAG_VERSION,
    MULTI_QUERY_RETRIEVER_TOOL,
    PipelineConfig,
    PipelineRegistry,
    UnknownPipelineError,
)


class StubPipeline:
    name = "stub"
    version = "1.0.0"

    async def run(self, state: QueryState) -> QueryState:
        return state


class StubTool:
    pass


def test_pipeline_registry_builds_default_and_explicit_pipeline() -> None:
    registry = PipelineRegistry(default_pipeline_name="stub")
    registry.register(
        config=PipelineConfig(name="stub", version="1.0.0", description="Test pipeline."),
        factory=StubPipeline,
    )
    registry.validate()

    assert registry.build().name == "stub"
    assert registry.build("stub").version == "1.0.0"
    assert registry.default_pipeline_name == "stub"


def test_pipeline_registry_rejects_duplicate_and_unknown_names() -> None:
    registry = PipelineRegistry(default_pipeline_name="stub")
    config = PipelineConfig(name="stub", version="1.0.0", description="Test pipeline.")
    registry.register(config=config, factory=StubPipeline)

    with pytest.raises(DuplicatePipelineError):
        registry.register(config=config, factory=StubPipeline)
    with pytest.raises(UnknownPipelineError, match="Available pipelines: stub"):
        registry.build("missing")


def test_tool_registry_resolves_named_capabilities() -> None:
    registry = ToolRegistry()
    tool = StubTool()
    config = ToolConfig(
        name="retriever.stub",
        kind="retriever",
        version="1.0.0",
        description="Test retriever.",
    )
    registry.register(config=config, implementation=tool)

    assert registry.resolve("retriever.stub", expected_type=StubTool) is tool
    assert registry.configs(kind="retriever") == (config,)

    with pytest.raises(DuplicateToolError):
        registry.register(config=config, implementation=tool)
    with pytest.raises(UnknownToolError):
        registry.resolve("retriever.missing")


def test_api_registry_exposes_baseline_pipeline_and_tools() -> None:
    settings = Settings(embedding_provider="hashing")
    tools = build_query_tool_registry(settings)
    pipelines = build_query_pipeline_registry(settings, tool_registry=tools)

    assert pipelines.default_pipeline_name == AGENTIC_RAG_NAME
    assert [config.name for config in pipelines.configs()] == [
        BASELINE_RAG_NAME,
        HYBRID_RAG_NAME,
        HYBRID_LLM_RERANK_RAG_NAME,
        HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        CONTEXTUAL_RAG_NAME,
        MULTI_QUERY_RAG_NAME,
        AGENTIC_RAG_NAME,
    ]
    assert {config.name for config in tools.configs()} == {
        QUERY_CLASSIFIER_TOOL,
        INFORMATION_NEED_DECOMPOSER_TOOL,
        RETRIEVAL_PLANNER_TOOL,
        EVIDENCE_GRADER_TOOL,
        BASELINE_RETRIEVER_TOOL,
        HYBRID_KEYWORD_RETRIEVER_TOOL,
        HYBRID_RETRIEVER_TOOL,
        HYBRID_LLM_RERANKER_TOOL,
        HYBRID_CROSS_ENCODER_RERANKER_TOOL,
        CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
        CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
        CONTEXTUAL_RETRIEVER_TOOL,
        MULTI_QUERY_GENERATOR_TOOL,
        MULTI_QUERY_RETRIEVER_TOOL,
        BASELINE_LLM_TOOL,
    }

    baseline_graph = build_query_graph(settings, pipeline_name=BASELINE_RAG_NAME)
    assert baseline_graph.name == BASELINE_RAG_NAME
    assert baseline_graph.version == BASELINE_RAG_VERSION

    hybrid_graph = build_query_graph(settings, pipeline_name=HYBRID_RAG_NAME)
    assert hybrid_graph.name == HYBRID_RAG_NAME
    assert hybrid_graph.version == HYBRID_RAG_VERSION

    llm_reranked_graph = build_query_graph(settings, pipeline_name=HYBRID_LLM_RERANK_RAG_NAME)
    assert llm_reranked_graph.name == HYBRID_LLM_RERANK_RAG_NAME
    assert llm_reranked_graph.version == HYBRID_LLM_RERANK_RAG_VERSION

    cross_encoder_graph = build_query_graph(settings, pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME)
    assert cross_encoder_graph.name == HYBRID_CROSS_ENCODER_RERANK_RAG_NAME
    assert cross_encoder_graph.version == HYBRID_CROSS_ENCODER_RERANK_RAG_VERSION

    contextual_graph = build_query_graph(settings, pipeline_name=CONTEXTUAL_RAG_NAME)
    assert contextual_graph.name == CONTEXTUAL_RAG_NAME
    assert contextual_graph.version == CONTEXTUAL_RAG_VERSION

    multi_query_graph = build_query_graph(settings, pipeline_name=MULTI_QUERY_RAG_NAME)
    assert multi_query_graph.name == MULTI_QUERY_RAG_NAME
    assert multi_query_graph.version == MULTI_QUERY_RAG_VERSION

    agentic_graph = build_query_graph(settings, pipeline_name=AGENTIC_RAG_NAME)
    assert agentic_graph.name == AGENTIC_RAG_NAME
    assert agentic_graph.version == AGENTIC_RAG_VERSION


def test_api_registry_rejects_invalid_configured_default() -> None:
    settings = Settings(default_query_pipeline="missing", embedding_provider="hashing")

    with pytest.raises(UnknownPipelineError, match="missing"):
        build_query_pipeline_registry(settings)


def test_settings_require_distinct_named_vectors() -> None:
    with pytest.raises(ValueError, match="must be different"):
        Settings(
            qdrant_original_vector_name="same",
            qdrant_contextual_vector_name="same",
            embedding_provider="hashing",
        )
