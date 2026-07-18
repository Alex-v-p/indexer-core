from __future__ import annotations

from typing import cast

from app.composition.providers import (
    build_contextual_keyword_store,
    build_cross_encoder_reranker,
    build_embedding_provider,
    build_keyword_store,
    build_language_model,
    build_ollama_reranker,
    build_vector_store,
)
from app.core.config import Settings
from packages.rag_core.query_understanding.classification import (
    LLMQueryClassifier,
    QUERY_CLASSIFIER_TOOL,
    QueryClassifier,
)
from packages.rag_core.agents.nodes import RetrievalPlanExecution
from packages.rag_core.agents.tools import ToolConfig, ToolRegistry
from packages.rag_core.pipelines import (
    AGENTIC_RAG_CONFIG,
    AGENTIC_RAG_NAME,
    BASELINE_LLM_TOOL,
    BASELINE_RAG_CONFIG,
    BASELINE_RAG_NAME,
    BASELINE_RAG_VERSION,
    BASELINE_RETRIEVER_TOOL,
    CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
    CONTEXTUAL_RAG_CONFIG,
    CONTEXTUAL_RAG_NAME,
    CONTEXTUAL_RAG_VERSION,
    CONTEXTUAL_RETRIEVER_TOOL,
    CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
    HYBRID_CROSS_ENCODER_RERANKER_TOOL,
    HYBRID_CROSS_ENCODER_RERANK_RAG_CONFIG,
    HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    HYBRID_CROSS_ENCODER_RERANK_RAG_VERSION,
    HYBRID_KEYWORD_RETRIEVER_TOOL,
    HYBRID_LLM_RERANKER_TOOL,
    HYBRID_LLM_RERANK_RAG_CONFIG,
    HYBRID_RAG_CONFIG,
    HYBRID_RAG_NAME,
    HYBRID_RAG_VERSION,
    HYBRID_RETRIEVER_TOOL,
    MULTI_QUERY_GENERATOR_TOOL,
    MULTI_QUERY_RAG_CONFIG,
    MULTI_QUERY_RAG_NAME,
    MULTI_QUERY_RAG_VERSION,
    MULTI_QUERY_RETRIEVER_TOOL,
    PipelineRegistry,
    RetrievalPipeline,
    build_agentic_rag_graph,
    build_baseline_rag_graph,
    build_contextual_rag_graph,
    build_hybrid_cross_encoder_rerank_rag_graph,
    build_hybrid_llm_rerank_rag_graph,
    build_hybrid_rag_graph,
    build_multi_query_rag_graph,
)
from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.decomposition import (
    INFORMATION_NEED_DECOMPOSER_TOOL,
    InformationNeedDecomposer,
    LLMInformationNeedDecomposer,
)
from packages.rag_core.query_understanding.planning import (
    RETRIEVAL_PLANNER_TOOL,
    RetrievalPlanner,
    RetrievalStrategy,
    RuleBasedRetrievalPlanner,
)
from packages.rag_core.retrieval import LLMQueryVariantGenerator
from packages.rag_core.retrieval.graders import (
    EVIDENCE_GRADER_TOOL,
    EvidenceGrader,
    LLMEvidenceGrader,
)
from packages.rag_core.retrieval.retry import (
    RETRIEVAL_RETRY_POLICY_TOOL,
    RetrievalRetryPolicy,
    RuleBasedRetrievalRetryPolicy,
)
from packages.rag_core.retrieval.rerankers import Reranker
from packages.rag_core.retrieval.retrievers import (
    HybridRetriever,
    KeywordRetriever,
    MultiQueryRetriever,
    Retriever,
    VectorRetriever,
)


def _build_hybrid_retriever(
    *,
    settings: Settings,
    vector_retriever: Retriever,
    keyword_retriever: Retriever,
) -> HybridRetriever:
    return HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        candidate_multiplier=settings.hybrid_candidate_multiplier,
        max_candidates=settings.hybrid_max_candidates,
        rrf_k=settings.hybrid_rrf_k,
        vector_weight=settings.hybrid_vector_weight,
        keyword_weight=settings.hybrid_keyword_weight,
    )


def build_query_tool_registry(settings: Settings) -> ToolRegistry:
    embedding_provider = build_embedding_provider(settings)
    llm_provider = build_language_model(settings)
    query_classifier = LLMQueryClassifier(
        llm_provider=llm_provider,
        fail_open=settings.query_classification_fail_open,
        max_rationale_chars=settings.query_classification_max_rationale_chars,
    )
    vector_store = build_vector_store(settings)
    vector_retriever = VectorRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        vector_name=settings.qdrant_original_vector_name,
    )
    keyword_retriever = KeywordRetriever(keyword_store=build_keyword_store(settings))
    hybrid_retriever = _build_hybrid_retriever(
        settings=settings,
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
    )

    contextual_vector_retriever = VectorRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        vector_name=settings.qdrant_contextual_vector_name,
    )
    contextual_keyword_retriever = KeywordRetriever(keyword_store=build_contextual_keyword_store(settings))
    contextual_retriever = _build_hybrid_retriever(
        settings=settings,
        vector_retriever=contextual_vector_retriever,
        keyword_retriever=contextual_keyword_retriever,
    )

    query_variant_generator = LLMQueryVariantGenerator(
        llm_provider=llm_provider,
        max_variant_chars=settings.multi_query_max_variant_chars,
    )
    information_need_decomposer = LLMInformationNeedDecomposer(
        llm_provider=llm_provider,
        fail_open=settings.information_need_decomposition_fail_open,
        max_information_needs=settings.information_need_max_count,
        max_need_chars=settings.information_need_max_chars,
        max_rationale_chars=settings.information_need_decomposition_max_rationale_chars,
    )
    retrieval_planner = RuleBasedRetrievalPlanner(
        baseline_pipeline_name=BASELINE_RAG_NAME,
        hybrid_pipeline_name=HYBRID_RAG_NAME,
        contextual_pipeline_name=CONTEXTUAL_RAG_NAME,
        multi_query_pipeline_name=MULTI_QUERY_RAG_NAME,
        rerank_pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        low_confidence_threshold=settings.retrieval_planning_low_confidence_threshold,
        contextual_available=settings.contextualization_enabled,
    )

    evidence_grader = LLMEvidenceGrader(
        llm_provider=llm_provider,
        fail_open=settings.evidence_grading_fail_open,
        relevance_threshold=settings.evidence_grading_relevance_threshold,
        information_need_support_threshold=settings.evidence_grading_information_need_support_threshold,
        max_chars_per_evidence=settings.evidence_grading_max_chars_per_evidence,
        max_rationale_chars=settings.evidence_grading_max_rationale_chars,
    )

    retry_policy = RuleBasedRetrievalRetryPolicy(
        pipeline_names={
            RetrievalStrategy.BASELINE: BASELINE_RAG_NAME,
            RetrievalStrategy.HYBRID: HYBRID_RAG_NAME,
            RetrievalStrategy.CONTEXTUAL: CONTEXTUAL_RAG_NAME,
            RetrievalStrategy.MULTI_QUERY: MULTI_QUERY_RAG_NAME,
            RetrievalStrategy.RERANK: HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        },
        max_retries=settings.retrieval_retry_max_retries,
        top_k_multiplier=settings.retrieval_retry_top_k_multiplier,
        max_top_k=settings.retrieval_retry_max_top_k,
        expand_query=settings.retrieval_retry_expand_query,
        max_query_chars=settings.retrieval_retry_max_query_chars,
    )

    multi_query_retriever = MultiQueryRetriever(
        query_variant_generator=query_variant_generator,
        retriever=hybrid_retriever,
        base_retrieval_strategy="hybrid",
        variant_count=settings.multi_query_variant_count,
        include_original=settings.multi_query_include_original,
        candidate_multiplier=settings.multi_query_candidate_multiplier,
        max_candidates_per_query=settings.multi_query_max_candidates_per_query,
        rrf_k=settings.multi_query_rrf_k,
        original_query_weight=settings.multi_query_original_query_weight,
        variant_query_weight=settings.multi_query_variant_query_weight,
        fail_open=settings.multi_query_fail_open,
    )

    registry = ToolRegistry()
    registry.register(
        config=ToolConfig(
            name=QUERY_CLASSIFIER_TOOL,
            kind="classifier",
            version="0.1.0",
            description=(
                "LLM-backed query classifier for factual lookup, broad explanation, comparison, and "
                "version-specific intent, with deterministic fail-open rules."
            ),
            metadata={
                "provider": "ollama",
                "model": settings.ollama_model,
                "query_types": (
                    "factual_lookup",
                    "broad_explanation",
                    "comparison",
                    "version_specific",
                ),
                "fail_open": settings.query_classification_fail_open,
            },
        ),
        implementation=query_classifier,
    )
    registry.register(
        config=ToolConfig(
            name=INFORMATION_NEED_DECOMPOSER_TOOL,
            kind="decomposer",
            version="0.1.0",
            description=(
                "LLM-backed query-understanding tool that extracts independently gradable information needs "
                "without selecting or executing a retrieval strategy."
            ),
            metadata={
                "provider": "ollama",
                "model": settings.ollama_model,
                "decomposer": information_need_decomposer.name,
                "fail_open": settings.information_need_decomposition_fail_open,
                "max_information_needs": settings.information_need_max_count,
                "max_information_need_chars": settings.information_need_max_chars,
            },
        ),
        implementation=information_need_decomposer,
    )
    registry.register(
        config=ToolConfig(
            name=RETRIEVAL_PLANNER_TOOL,
            kind="planner",
            version="0.3.0",
            description=(
                "Retrieval planner that consumes independent query classification and information-need "
                "decomposition results before selecting baseline, hybrid, contextual, multi-query, or reranked retrieval."
            ),
            metadata={
                "planner": retrieval_planner.name,
                "low_confidence_threshold": settings.retrieval_planning_low_confidence_threshold,
                "contextual_available": settings.contextualization_enabled,
                "rerank_pipeline": HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
                "inputs": ("query_classification", "information_need_decomposition"),
            },
        ),
        implementation=retrieval_planner,
    )
    registry.register(
        config=ToolConfig(
            name=EVIDENCE_GRADER_TOOL,
            kind="grader",
            version="0.2.0",
            description=(
                "LLM-backed evidence grader that scores every chunk and every planned information need, "
                "then blocks generation until all required needs are supported."
            ),
            metadata={
                "provider": "ollama",
                "model": settings.ollama_model,
                "fail_open": settings.evidence_grading_fail_open,
                "relevance_threshold": settings.evidence_grading_relevance_threshold,
                "information_need_support_threshold": settings.evidence_grading_information_need_support_threshold,
                "max_chars_per_evidence": settings.evidence_grading_max_chars_per_evidence,
            },
        ),
        implementation=evidence_grader,
    )
    registry.register(
        config=ToolConfig(
            name=RETRIEVAL_RETRY_POLICY_TOOL,
            kind="retry_policy",
            version="0.1.0",
            description=(
                "Deterministic bounded retry policy that expands unresolved retrieval queries, increases top-k, "
                "and escalates across registered retrieval pipelines after weak or missing evidence."
            ),
            metadata={
                "policy": retry_policy.name,
                "max_retries": settings.retrieval_retry_max_retries,
                "top_k_multiplier": settings.retrieval_retry_top_k_multiplier,
                "max_top_k": settings.retrieval_retry_max_top_k,
                "expand_query": settings.retrieval_retry_expand_query,
                "fallback_order": ("baseline", "hybrid", "multi_query", "rerank"),
            },
        ),
        implementation=retry_policy,
    )
    registry.register(
        config=ToolConfig(
            name=BASELINE_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="Dense-vector retriever backed by the configured embedding provider and Qdrant.",
            metadata={"strategy": "dense_vector", "collection": settings.qdrant_collection, "vector_name": settings.qdrant_original_vector_name},
        ),
        implementation=vector_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=HYBRID_KEYWORD_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="BM25 keyword retriever built from searchable Qdrant chunk payloads.",
            metadata={
                "strategy": "keyword_bm25",
                "collection": settings.qdrant_collection,
                "search_text_field": "text",
                "k1": settings.keyword_bm25_k1,
                "b": settings.keyword_bm25_b,
            },
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
            name=MULTI_QUERY_GENERATOR_TOOL,
            kind="query_generator",
            version="0.1.0",
            description="LLM-backed generator for intent-preserving retrieval query variants.",
            metadata={
                "provider": "ollama",
                "model": settings.ollama_model,
                "variant_count": settings.multi_query_variant_count,
                "max_variant_chars": settings.multi_query_max_variant_chars,
            },
        ),
        implementation=query_variant_generator,
    )
    registry.register(
        config=ToolConfig(
            name=MULTI_QUERY_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="Multi-query retriever using hybrid search per query and weighted RRF across variants.",
            metadata={
                "strategy": "multi_query_hybrid",
                "base_retriever": HYBRID_RETRIEVER_TOOL,
                "variant_count": settings.multi_query_variant_count,
                "include_original": settings.multi_query_include_original,
                "candidate_multiplier": settings.multi_query_candidate_multiplier,
                "max_candidates_per_query": settings.multi_query_max_candidates_per_query,
                "rrf_k": settings.multi_query_rrf_k,
                "original_query_weight": settings.multi_query_original_query_weight,
                "variant_query_weight": settings.multi_query_variant_query_weight,
                "fail_open": settings.multi_query_fail_open,
            },
        ),
        implementation=multi_query_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="Dense-vector retriever over document-aware contextualized chunk embeddings.",
            metadata={
                "strategy": "contextual_dense_vector",
                "collection": settings.qdrant_collection,
                "contextualization_enabled": settings.contextualization_enabled,
                "vector_name": settings.qdrant_contextual_vector_name,
                "evidence_text": "original_chunk",
            },
        ),
        implementation=contextual_vector_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="BM25 retrieval over contextualized chunk text with original chunk evidence returned.",
            metadata={
                "strategy": "contextual_keyword_bm25",
                "collection": settings.qdrant_collection,
                "contextualization_enabled": settings.contextualization_enabled,
                "search_text_field": "contextualized_text",
                "evidence_text_field": "text",
                "k1": settings.keyword_bm25_k1,
                "b": settings.keyword_bm25_b,
            },
        ),
        implementation=contextual_keyword_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=CONTEXTUAL_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description="Hybrid vector/BM25 retrieval over contextualized chunk representations.",
            metadata={
                "strategy": "contextual_hybrid",
                "fusion": "weighted_reciprocal_rank_fusion",
                "collection": settings.qdrant_collection,
                "contextualization_enabled": settings.contextualization_enabled,
                "rrf_k": settings.hybrid_rrf_k,
                "candidate_multiplier": settings.hybrid_candidate_multiplier,
                "vector_weight": settings.hybrid_vector_weight,
                "keyword_weight": settings.hybrid_keyword_weight,
            },
        ),
        implementation=contextual_retriever,
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
        implementation=llm_provider,
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
        config=AGENTIC_RAG_CONFIG,
        factory=lambda: build_agentic_rag_graph(
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
            information_need_decomposer=cast(
                InformationNeedDecomposer,
                tools.resolve(INFORMATION_NEED_DECOMPOSER_TOOL),
            ),
            retrieval_planner=cast(RetrievalPlanner, tools.resolve(RETRIEVAL_PLANNER_TOOL)),
            executions={
                BASELINE_RAG_NAME: RetrievalPlanExecution(
                    pipeline_name=BASELINE_RAG_NAME,
                    pipeline_version=BASELINE_RAG_VERSION,
                    strategy=RetrievalStrategy.BASELINE,
                    retriever=cast(Retriever, tools.resolve(BASELINE_RETRIEVER_TOOL)),
                ),
                HYBRID_RAG_NAME: RetrievalPlanExecution(
                    pipeline_name=HYBRID_RAG_NAME,
                    pipeline_version=HYBRID_RAG_VERSION,
                    strategy=RetrievalStrategy.HYBRID,
                    retriever=cast(Retriever, tools.resolve(HYBRID_RETRIEVER_TOOL)),
                ),
                CONTEXTUAL_RAG_NAME: RetrievalPlanExecution(
                    pipeline_name=CONTEXTUAL_RAG_NAME,
                    pipeline_version=CONTEXTUAL_RAG_VERSION,
                    strategy=RetrievalStrategy.CONTEXTUAL,
                    retriever=cast(Retriever, tools.resolve(CONTEXTUAL_RETRIEVER_TOOL)),
                ),
                MULTI_QUERY_RAG_NAME: RetrievalPlanExecution(
                    pipeline_name=MULTI_QUERY_RAG_NAME,
                    pipeline_version=MULTI_QUERY_RAG_VERSION,
                    strategy=RetrievalStrategy.MULTI_QUERY,
                    retriever=cast(Retriever, tools.resolve(MULTI_QUERY_RETRIEVER_TOOL)),
                ),
                HYBRID_CROSS_ENCODER_RERANK_RAG_NAME: RetrievalPlanExecution(
                    pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
                    pipeline_version=HYBRID_CROSS_ENCODER_RERANK_RAG_VERSION,
                    strategy=RetrievalStrategy.RERANK,
                    retriever=cast(Retriever, tools.resolve(HYBRID_RETRIEVER_TOOL)),
                    reranker=cast(Reranker, tools.resolve(HYBRID_CROSS_ENCODER_RERANKER_TOOL)),
                    candidate_multiplier=settings.rerank_candidate_multiplier,
                    max_candidates=settings.rerank_max_candidates,
                ),
            },
            evidence_grader=cast(EvidenceGrader, tools.resolve(EVIDENCE_GRADER_TOOL)),
            retry_policy=cast(RetrievalRetryPolicy, tools.resolve(RETRIEVAL_RETRY_POLICY_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
        ),
    )
    registry.validate()
    for pipeline_config in registry.configs():
        for tool_name in pipeline_config.tool_names:
            tools.resolve(tool_name)
    return registry


def build_query_graph(settings: Settings, *, pipeline_name: str | None = None) -> RetrievalPipeline:
    return build_query_pipeline_registry(settings).build(pipeline_name)
