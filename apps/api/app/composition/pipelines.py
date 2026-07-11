from __future__ import annotations

from typing import cast

from app.composition.providers import (
    build_cross_encoder_reranker,
    build_embedding_provider,
    build_keyword_store,
    build_language_model,
    build_ollama_reranker,
    build_vector_store,
)
from app.core.config import Settings
from packages.rag_core.agents.tools import ToolConfig, ToolRegistry
from packages.rag_core.pipelines import (
    BASELINE_LLM_TOOL,
    BASELINE_RAG_CONFIG,
    BASELINE_RETRIEVER_TOOL,
    HYBRID_CROSS_ENCODER_RERANKER_TOOL,
    HYBRID_CROSS_ENCODER_RERANK_RAG_CONFIG,
    HYBRID_KEYWORD_RETRIEVER_TOOL,
    HYBRID_RAG_CONFIG,
    HYBRID_LLM_RERANKER_TOOL,
    HYBRID_LLM_RERANK_RAG_CONFIG,
    HYBRID_RETRIEVER_TOOL,
    PipelineRegistry,
    RetrievalPipeline,
    build_baseline_rag_graph,
    build_hybrid_cross_encoder_rerank_rag_graph,
    build_hybrid_rag_graph,
    build_hybrid_llm_rerank_rag_graph,
)
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.rerankers import Reranker
from packages.rag_core.retrieval.retrievers import HybridRetriever, KeywordRetriever, Retriever, VectorRetriever


def build_query_tool_registry(settings: Settings) -> ToolRegistry:
    vector_retriever = VectorRetriever(
        embedding_provider=build_embedding_provider(settings),
        vector_store=build_vector_store(settings),
    )
    keyword_retriever = KeywordRetriever(keyword_store=build_keyword_store(settings))
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        candidate_multiplier=settings.hybrid_candidate_multiplier,
        max_candidates=settings.hybrid_max_candidates,
        rrf_k=settings.hybrid_rrf_k,
        vector_weight=settings.hybrid_vector_weight,
        keyword_weight=settings.hybrid_keyword_weight,
    )

    registry = ToolRegistry()
    registry.register(
        config=ToolConfig(
            name=BASELINE_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="Dense-vector retriever backed by the configured embedding provider and Qdrant.",
            metadata={"strategy": "dense_vector"},
        ),
        implementation=vector_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=HYBRID_KEYWORD_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="BM25 keyword retriever built from searchable Qdrant chunk payloads.",
            metadata={"strategy": "keyword_bm25", "k1": settings.keyword_bm25_k1, "b": settings.keyword_bm25_b},
        ),
        implementation=keyword_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=HYBRID_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="Hybrid retriever that fuses dense and keyword rankings with weighted RRF.",
            metadata={
                "strategy": "hybrid",
                "fusion": "weighted_reciprocal_rank_fusion",
                "rrf_k": settings.hybrid_rrf_k,
                "candidate_multiplier": settings.hybrid_candidate_multiplier,
                "vector_weight": settings.hybrid_vector_weight,
                "keyword_weight": settings.hybrid_keyword_weight,
            },
        ),
        implementation=hybrid_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=HYBRID_LLM_RERANKER_TOOL,
            kind="reranker",
            version="0.2.0",
            description=(
                "Query-aware pointwise relevance reranker backed by Ollama structured output, "
                "with incomplete-response retries and original-rank fallback."
            ),
            metadata={
                "provider": "ollama",
                "model": settings.ollama_rerank_model,
                "batch_size": settings.rerank_batch_size,
                "max_chars_per_candidate": settings.rerank_max_chars_per_candidate,
                "max_attempts": settings.ollama_rerank_max_attempts,
                "fallback_to_original_rank": settings.ollama_rerank_fallback_to_original_rank,
            },
        ),
        implementation=build_ollama_reranker(settings),
    )
    registry.register(
        config=ToolConfig(
            name=HYBRID_CROSS_ENCODER_RERANKER_TOOL,
            kind="reranker",
            version="0.1.0",
            description="Local query/passage cross-encoder reranker loaded lazily through Sentence Transformers.",
            metadata={
                "provider": "cross_encoder",
                "model": settings.cross_encoder_model,
                "revision": settings.cross_encoder_model_revision,
                "model_path": settings.cross_encoder_model_path,
                "local_files_only": settings.cross_encoder_local_files_only,
                "batch_size": settings.cross_encoder_batch_size,
                "max_length": settings.cross_encoder_max_length,
                "device": settings.cross_encoder_device,
            },
        ),
        implementation=build_cross_encoder_reranker(settings),
    )
    registry.register(
        config=ToolConfig(
            name=BASELINE_LLM_TOOL,
            kind="generator",
            version="0.1.0",
            description="Configured LLM provider used by citation-aware answer generation.",
            metadata={"provider": "ollama"},
        ),
        implementation=build_language_model(settings),
    )
    return registry


def build_query_pipeline_registry(
    settings: Settings,
    *,
    tool_registry: ToolRegistry | None = None,
) -> PipelineRegistry:
    tools = tool_registry or build_query_tool_registry(settings)
    registry = PipelineRegistry(default_pipeline_name=settings.default_query_pipeline)
    registry.register(
        config=BASELINE_RAG_CONFIG,
        factory=lambda: build_baseline_rag_graph(
            retriever=cast(Retriever, tools.resolve(BASELINE_RETRIEVER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
        ),
    )
    registry.register(
        config=HYBRID_RAG_CONFIG,
        factory=lambda: build_hybrid_rag_graph(
            retriever=cast(Retriever, tools.resolve(HYBRID_RETRIEVER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
        ),
    )
    registry.register(
        config=HYBRID_LLM_RERANK_RAG_CONFIG,
        factory=lambda: build_hybrid_llm_rerank_rag_graph(
            retriever=cast(Retriever, tools.resolve(HYBRID_RETRIEVER_TOOL)),
            reranker=cast(Reranker, tools.resolve(HYBRID_LLM_RERANKER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
            candidate_multiplier=settings.rerank_candidate_multiplier,
            max_candidates=settings.rerank_max_candidates,
        ),
    )
    registry.register(
        config=HYBRID_CROSS_ENCODER_RERANK_RAG_CONFIG,
        factory=lambda: build_hybrid_cross_encoder_rerank_rag_graph(
            retriever=cast(Retriever, tools.resolve(HYBRID_RETRIEVER_TOOL)),
            reranker=cast(Reranker, tools.resolve(HYBRID_CROSS_ENCODER_RERANKER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
            candidate_multiplier=settings.rerank_candidate_multiplier,
            max_candidates=settings.rerank_max_candidates,
        ),
    )
    registry.validate()
    for pipeline_config in registry.configs():
        for tool_name in pipeline_config.tool_names:
            tools.resolve(tool_name)
    return registry


def build_query_graph(settings: Settings, *, pipeline_name: str | None = None) -> RetrievalPipeline:
    return build_query_pipeline_registry(settings).build(pipeline_name)
