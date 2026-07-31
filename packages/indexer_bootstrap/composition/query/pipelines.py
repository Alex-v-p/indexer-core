from __future__ import annotations

from typing import cast

from packages.indexer_bootstrap.config import Settings
from packages.rag_core.agents.tools import ToolRegistry
from packages.rag_core.pipelines import (
    BASELINE_LLM_TOOL,
    BASELINE_RAG_CONFIG,
    BASELINE_RETRIEVER_TOOL,
    CONTEXTUAL_RAG_CONFIG,
    CONTEXTUAL_RETRIEVER_TOOL,
    HIERARCHICAL_RAG_CONFIG,
    HIERARCHICAL_RETRIEVER_TOOL,
    HYBRID_CROSS_ENCODER_RERANKER_TOOL,
    HYBRID_CROSS_ENCODER_RERANK_RAG_CONFIG,
    HYBRID_LLM_RERANKER_TOOL,
    HYBRID_LLM_RERANK_RAG_CONFIG,
    HYBRID_RAG_CONFIG,
    HYBRID_RETRIEVER_TOOL,
    MULTI_QUERY_RAG_CONFIG,
    MULTI_QUERY_RETRIEVER_TOOL,
    PipelineRegistry,
    build_baseline_rag_graph,
    build_contextual_rag_graph,
    build_hierarchical_rag_graph,
    build_hybrid_cross_encoder_rerank_rag_graph,
    build_hybrid_llm_rerank_rag_graph,
    build_hybrid_rag_graph,
    build_multi_query_rag_graph,
)
from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL, QueryClassifier
from packages.rag_core.retrieval.rerankers import Reranker
from packages.rag_core.retrieval.retrievers import Retriever


def register_fixed_query_pipelines(
    registry: PipelineRegistry,
    *,
    settings: Settings,
    tools: ToolRegistry,
) -> None:
    """Register the fixed, single-strategy query pipeline family."""

    registry.register(
        config=BASELINE_RAG_CONFIG,
        factory=lambda: build_baseline_rag_graph(
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
            retriever=cast(Retriever, tools.resolve(BASELINE_RETRIEVER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
        ),
    )
    registry.register(
        config=HYBRID_RAG_CONFIG,
        factory=lambda: build_hybrid_rag_graph(
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
            retriever=cast(Retriever, tools.resolve(HYBRID_RETRIEVER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
        ),
    )
    registry.register(
        config=HYBRID_LLM_RERANK_RAG_CONFIG,
        factory=lambda: build_hybrid_llm_rerank_rag_graph(
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
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
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
            retriever=cast(Retriever, tools.resolve(HYBRID_RETRIEVER_TOOL)),
            reranker=cast(Reranker, tools.resolve(HYBRID_CROSS_ENCODER_RERANKER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
            candidate_multiplier=settings.rerank_candidate_multiplier,
            max_candidates=settings.rerank_max_candidates,
        ),
    )
    registry.register(
        config=CONTEXTUAL_RAG_CONFIG,
        factory=lambda: build_contextual_rag_graph(
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
            retriever=cast(Retriever, tools.resolve(CONTEXTUAL_RETRIEVER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
        ),
    )
    registry.register(
        config=MULTI_QUERY_RAG_CONFIG,
        factory=lambda: build_multi_query_rag_graph(
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
            retriever=cast(Retriever, tools.resolve(MULTI_QUERY_RETRIEVER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
        ),
    )
    registry.register(
        config=HIERARCHICAL_RAG_CONFIG,
        factory=lambda: build_hierarchical_rag_graph(
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
            retriever=cast(Retriever, tools.resolve(HIERARCHICAL_RETRIEVER_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
        ),
    )
