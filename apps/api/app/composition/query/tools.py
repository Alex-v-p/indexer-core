from __future__ import annotations

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
from packages.rag_core.agents.tools import ToolConfig, ToolRegistry
from packages.rag_core.pipelines import (
    BASELINE_LLM_TOOL,
    BASELINE_RAG_NAME,
    BASELINE_RETRIEVER_TOOL,
    CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
    CONTEXTUAL_RAG_NAME,
    CONTEXTUAL_RETRIEVER_TOOL,
    CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
    HIERARCHICAL_RAG_NAME,
    HIERARCHICAL_RETRIEVER_TOOL,
    HYBRID_CROSS_ENCODER_RERANKER_TOOL,
    HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    HYBRID_KEYWORD_RETRIEVER_TOOL,
    HYBRID_LLM_RERANKER_TOOL,
    HYBRID_RAG_NAME,
    HYBRID_RETRIEVER_TOOL,
    MULTI_QUERY_GENERATOR_TOOL,
    MULTI_QUERY_RAG_NAME,
    MULTI_QUERY_RETRIEVER_TOOL,
)
from packages.rag_core.query_understanding.classification import (
    LLMQueryClassifier,
    QUERY_CLASSIFIER_TOOL,
)
from packages.rag_core.query_understanding.decomposition import (
    INFORMATION_NEED_DECOMPOSER_TOOL,
    LLMInformationNeedDecomposer,
)
from packages.rag_core.query_understanding.planning import (
    RETRIEVAL_PLANNER_TOOL,
    RetrievalStrategy,
    RuleBasedRetrievalPlanner,
)
from packages.rag_core.retrieval import LLMQueryVariantGenerator
from packages.rag_core.retrieval.arbitration import (
    EVIDENCE_ARBITRATOR_TOOL,
    LLMQuestionEvidenceArbitrator,
)
from packages.rag_core.retrieval.document_selection import (
    DOCUMENT_CANDIDATE_SELECTOR_TOOL,
    PRIMARY_DOCUMENT_DETECTOR_TOOL,
    DocumentBalancedCandidateSelector,
    DocumentCandidateSelector,
    NoPrimaryDocumentDetector,
    PassthroughDocumentCandidateSelector,
    PrimaryDocumentDetector,
    RuleBasedPrimaryDocumentDetector,
)
from packages.rag_core.retrieval.graders import EVIDENCE_GRADER_TOOL, LLMEvidenceGrader
from packages.rag_core.retrieval.retry import (
    RETRIEVAL_RETRY_POLICY_TOOL,
    RuleBasedRetrievalRetryPolicy,
)
from packages.rag_core.retrieval.retrievers import (
    HierarchicalRetriever,
    HierarchicalRetrieverConfig,
    HybridRetriever,
    KeywordRetriever,
    MultiQueryRetriever,
    Retriever,
    VectorRetriever,
    VersionAwareRetriever,
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
        timezone_name=settings.temporal_query_timezone,
        max_repair_attempts=settings.structured_output_max_repair_attempts,
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
    hierarchical_retriever = HierarchicalRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        hierarchy_vector_name=settings.qdrant_hierarchy_vector_name,
        chunk_vector_name=(
            settings.qdrant_contextual_vector_name
            if settings.contextualization_enabled
            else settings.qdrant_original_vector_name
        ),
        fallback_chunk_vector_name=(
            settings.qdrant_original_vector_name if settings.contextualization_enabled else None
        ),
        config=HierarchicalRetrieverConfig(
            document_candidates=settings.hierarchical_document_candidates,
            section_candidates=settings.hierarchical_section_candidates,
            chunk_candidate_multiplier=settings.hierarchical_chunk_candidate_multiplier,
            max_chunk_candidates=settings.hierarchical_max_chunk_candidates,
        ),
    )

    query_variant_generator = LLMQueryVariantGenerator(
        llm_provider=llm_provider,
        max_variant_chars=settings.multi_query_max_variant_chars,
        max_repair_attempts=settings.structured_output_max_repair_attempts,
    )
    information_need_decomposer = LLMInformationNeedDecomposer(
        llm_provider=llm_provider,
        fail_open=settings.information_need_decomposition_fail_open,
        max_information_needs=settings.information_need_max_count,
        max_need_chars=settings.information_need_max_chars,
        max_rationale_chars=settings.information_need_decomposition_max_rationale_chars,
        max_repair_attempts=settings.structured_output_max_repair_attempts,
    )
    retrieval_planner = RuleBasedRetrievalPlanner(
        baseline_pipeline_name=BASELINE_RAG_NAME,
        hybrid_pipeline_name=HYBRID_RAG_NAME,
        contextual_pipeline_name=CONTEXTUAL_RAG_NAME,
        multi_query_pipeline_name=MULTI_QUERY_RAG_NAME,
        rerank_pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        hierarchical_pipeline_name=HIERARCHICAL_RAG_NAME,
        low_confidence_threshold=settings.retrieval_planning_low_confidence_threshold,
        contextual_available=settings.contextualization_enabled,
        hierarchical_available=settings.hierarchical_indexing_enabled,
        top_k_multiplier=settings.retrieval_retry_top_k_multiplier,
        max_top_k=settings.retrieval_retry_max_top_k,
        expand_query=settings.retrieval_retry_expand_query,
        max_query_chars=settings.retrieval_retry_max_query_chars,
    )

    evidence_grader = LLMEvidenceGrader(
        llm_provider=llm_provider,
        fail_open=settings.evidence_grading_fail_open,
        relevance_threshold=settings.evidence_grading_relevance_threshold,
        information_need_support_threshold=settings.evidence_grading_information_need_support_threshold,
        max_chars_per_evidence=settings.evidence_grading_max_chars_per_evidence,
        max_rationale_chars=settings.evidence_grading_max_rationale_chars,
        max_repair_attempts=settings.structured_output_max_repair_attempts,
    )

    evidence_arbitrator = LLMQuestionEvidenceArbitrator(
        llm_provider=llm_provider,
        fail_open=settings.evidence_arbitration_fail_open,
        relevance_threshold=settings.evidence_arbitration_relevance_threshold,
        information_need_support_threshold=settings.evidence_arbitration_information_need_support_threshold,
        max_chars_per_evidence=settings.evidence_arbitration_max_chars_per_evidence,
        max_rationale_chars=settings.evidence_arbitration_max_rationale_chars,
        max_repair_attempts=settings.structured_output_max_repair_attempts,
    )

    primary_document_detector: PrimaryDocumentDetector = (
        RuleBasedPrimaryDocumentDetector(
            min_score=settings.primary_document_detection_min_score,
            min_margin=settings.primary_document_detection_min_margin,
            replacement_margin=settings.primary_document_detection_replacement_margin,
        )
        if settings.primary_document_detection_enabled
        else NoPrimaryDocumentDetector()
    )
    document_candidate_selector: DocumentCandidateSelector = (
        DocumentBalancedCandidateSelector(
            candidate_multiplier=settings.document_balancing_candidate_multiplier,
            max_candidates=settings.document_balancing_max_candidates,
            primary_min_share=settings.document_balancing_primary_min_share,
            primary_max_share=settings.document_balancing_primary_max_share,
            secondary_max_share=settings.document_balancing_secondary_max_share,
            unpreferred_max_share=settings.document_balancing_unpreferred_max_share,
        )
        if settings.document_balancing_enabled
        else PassthroughDocumentCandidateSelector()
    )

    retry_pipeline_names = {
        RetrievalStrategy.BASELINE: BASELINE_RAG_NAME,
        RetrievalStrategy.HYBRID: HYBRID_RAG_NAME,
        RetrievalStrategy.MULTI_QUERY: MULTI_QUERY_RAG_NAME,
        RetrievalStrategy.RERANK: HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    }
    if settings.contextualization_enabled:
        retry_pipeline_names[RetrievalStrategy.CONTEXTUAL] = CONTEXTUAL_RAG_NAME
    if settings.hierarchical_indexing_enabled:
        retry_pipeline_names[RetrievalStrategy.HIERARCHICAL] = HIERARCHICAL_RAG_NAME

    retry_policy = RuleBasedRetrievalRetryPolicy(
        pipeline_names=retry_pipeline_names,
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

    version_candidate_limit = max(settings.hybrid_max_candidates, settings.retrieval_retry_max_top_k * 4)
    baseline_version_aware_retriever = VersionAwareRetriever(
        vector_retriever,
        max_candidates=version_candidate_limit,
    )
    keyword_version_aware_retriever = VersionAwareRetriever(
        keyword_retriever,
        max_candidates=version_candidate_limit,
    )
    hybrid_version_aware_retriever = VersionAwareRetriever(
        hybrid_retriever,
        max_candidates=version_candidate_limit,
    )
    contextual_vector_version_aware_retriever = VersionAwareRetriever(
        contextual_vector_retriever,
        max_candidates=version_candidate_limit,
    )
    contextual_keyword_version_aware_retriever = VersionAwareRetriever(
        contextual_keyword_retriever,
        max_candidates=version_candidate_limit,
    )
    contextual_version_aware_retriever = VersionAwareRetriever(
        contextual_retriever,
        max_candidates=version_candidate_limit,
    )
    hierarchical_version_aware_retriever = VersionAwareRetriever(
        hierarchical_retriever,
        max_candidates=version_candidate_limit,
    )
    multi_query_version_aware_retriever = VersionAwareRetriever(
        multi_query_retriever,
        max_candidates=version_candidate_limit,
    )

    registry = ToolRegistry()
    registry.register(
        config=ToolConfig(
            name=QUERY_CLASSIFIER_TOOL,
            kind="classifier",
            version="0.2.0",
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
                "generation_profile": "deterministic_json_schema",
                "temperature": settings.ollama_query_temperature,
                "seed": settings.ollama_query_seed,
                "max_repair_attempts": settings.structured_output_max_repair_attempts,
            },
        ),
        implementation=query_classifier,
    )
    registry.register(
        config=ToolConfig(
            name=INFORMATION_NEED_DECOMPOSER_TOOL,
            kind="decomposer",
            version="0.2.0",
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
                "generation_profile": "deterministic_json_schema",
                "temperature": settings.ollama_query_temperature,
                "seed": settings.ollama_query_seed,
                "max_repair_attempts": settings.structured_output_max_repair_attempts,
            },
        ),
        implementation=information_need_decomposer,
    )
    registry.register(
        config=ToolConfig(
            name=RETRIEVAL_PLANNER_TOOL,
            kind="planner",
            version="0.5.0",
            description=(
                "Per-information-need retrieval planner that selects an independent query, pipeline, top-k, and "
                "fallback attempt from that item's classification and grader history."
            ),
            metadata={
                "planner": retrieval_planner.name,
                "low_confidence_threshold": settings.retrieval_planning_low_confidence_threshold,
                "contextual_available": settings.contextualization_enabled,
                "rerank_pipeline": HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
                "hierarchical_pipeline": HIERARCHICAL_RAG_NAME,
                "inputs": ("information_need", "information_need_classification", "previous_grade", "attempt_history"),
                "planning_scope": "per_information_need",
            },
        ),
        implementation=retrieval_planner,
    )
    registry.register(
        config=ToolConfig(
            name=EVIDENCE_GRADER_TOOL,
            kind="grader",
            version="0.3.0",
            description=(
                "LLM-backed evidence grader that scores every chunk and one active information need "
                "per subgraph pass, allowing complete answers when all required needs are supported "
                "and explicit partial answers otherwise."
            ),
            metadata={
                "provider": "ollama",
                "model": settings.ollama_model,
                "fail_open": settings.evidence_grading_fail_open,
                "relevance_threshold": settings.evidence_grading_relevance_threshold,
                "information_need_support_threshold": settings.evidence_grading_information_need_support_threshold,
                "max_chars_per_evidence": settings.evidence_grading_max_chars_per_evidence,
                "generation_profile": "deterministic_json_schema",
                "temperature": settings.ollama_query_temperature,
                "seed": settings.ollama_query_seed,
                "max_repair_attempts": settings.structured_output_max_repair_attempts,
            },
        ),
        implementation=evidence_grader,
    )
    registry.register(
        config=ToolConfig(
            name=PRIMARY_DOCUMENT_DETECTOR_TOOL,
            kind="document_detector",
            version="0.1.0",
            description=(
                "Deterministic soft primary-document detector that scores directly graded evidence by "
                "question relevance and document-name overlap without creating a hard metadata filter."
            ),
            metadata={
                "enabled": settings.primary_document_detection_enabled,
                "min_score": settings.primary_document_detection_min_score,
                "min_margin": settings.primary_document_detection_min_margin,
                "replacement_margin": settings.primary_document_detection_replacement_margin,
                "semantics": "soft_preference_not_filter",
            },
        ),
        implementation=primary_document_detector,
    )
    registry.register(
        config=ToolConfig(
            name=DOCUMENT_CANDIDATE_SELECTOR_TOOL,
            kind="candidate_selector",
            version="0.1.0",
            description=(
                "Document-aware candidate selector that expands retrieval, reserves a majority for the learned "
                "primary document, and keeps bounded slots for supporting documents."
            ),
            metadata={
                "enabled": settings.document_balancing_enabled,
                "candidate_multiplier": settings.document_balancing_candidate_multiplier,
                "max_candidates": settings.document_balancing_max_candidates,
                "primary_min_share": settings.document_balancing_primary_min_share,
                "primary_max_share": settings.document_balancing_primary_max_share,
                "secondary_max_share": settings.document_balancing_secondary_max_share,
                "unpreferred_max_share": settings.document_balancing_unpreferred_max_share,
            },
        ),
        implementation=document_candidate_selector,
    )
    registry.register(
        config=ToolConfig(
            name=EVIDENCE_ARBITRATOR_TOOL,
            kind="arbiter",
            version="0.2.0",
            description=(
                "Strict original-question-level evidence arbiter that removes accumulated chunks which do not "
                "directly support a final answer claim and cannot upgrade prior per-need support decisions."
            ),
            metadata={
                "provider": "ollama",
                "model": settings.ollama_model,
                "fail_open": settings.evidence_arbitration_fail_open,
                "relevance_threshold": settings.evidence_arbitration_relevance_threshold,
                "information_need_support_threshold": settings.evidence_arbitration_information_need_support_threshold,
                "max_chars_per_evidence": settings.evidence_arbitration_max_chars_per_evidence,
                "scope": "original_question_final_generation_gate",
                "generation_profile": "deterministic_json_schema",
                "temperature": settings.ollama_query_temperature,
                "seed": settings.ollama_query_seed,
                "max_repair_attempts": settings.structured_output_max_repair_attempts,
            },
        ),
        implementation=evidence_arbitrator,
    )
    registry.register(
        config=ToolConfig(
            name=RETRIEVAL_RETRY_POLICY_TOOL,
            kind="retry_policy",
            version="0.3.0",
            description=(
                "Deterministic controller that routes each information need to retry, reclassification, supported "
                "completion, or exhausted completion while enforcing per-item and query-level budgets."
            ),
            metadata={
                "policy": retry_policy.name,
                "max_retries": settings.retrieval_retry_max_retries,
                "top_k_multiplier": settings.retrieval_retry_top_k_multiplier,
                "max_top_k": settings.retrieval_retry_max_top_k,
                "expand_query": settings.retrieval_retry_expand_query,
                "max_total_attempts": settings.retrieval_retry_max_total_attempts,
                "max_reclassifications": settings.retrieval_retry_max_reclassifications,
                "max_accumulated_evidence": settings.retrieval_retry_max_accumulated_evidence,
                "scope": "per_information_need",
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
            metadata={
                "strategy": "dense_vector",
                "collection": settings.qdrant_collection,
                "vector_name": settings.qdrant_original_vector_name,
            },
        ),
        implementation=baseline_version_aware_retriever,
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
        implementation=keyword_version_aware_retriever,
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
        implementation=hybrid_version_aware_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=MULTI_QUERY_GENERATOR_TOOL,
            kind="query_generator",
            version="0.2.0",
            description="LLM-backed generator for intent-preserving retrieval query variants.",
            metadata={
                "provider": "ollama",
                "model": settings.ollama_model,
                "variant_count": settings.multi_query_variant_count,
                "max_variant_chars": settings.multi_query_max_variant_chars,
                "generation_profile": "deterministic_json_schema",
                "temperature": settings.ollama_query_temperature,
                "seed": settings.ollama_query_seed,
                "max_repair_attempts": settings.structured_output_max_repair_attempts,
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
        implementation=multi_query_version_aware_retriever,
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
        implementation=contextual_vector_version_aware_retriever,
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
        implementation=contextual_keyword_version_aware_retriever,
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
        implementation=contextual_version_aware_retriever,
    )
    registry.register(
        config=ToolConfig(
            name=HIERARCHICAL_RETRIEVER_TOOL,
            kind="retriever",
            version="0.1.0",
            description=(
                "Three-stage retriever that selects document summaries, then semantic-section summaries, "
                "then precise source chunks from the selected hierarchy branches."
            ),
            metadata={
                "strategy": "hierarchical_document_section_chunk",
                "collection": settings.qdrant_collection,
                "hierarchy_vector_name": settings.qdrant_hierarchy_vector_name,
                "chunk_vector_name": (
                    settings.qdrant_contextual_vector_name
                    if settings.contextualization_enabled
                    else settings.qdrant_original_vector_name
                ),
                "document_candidates": settings.hierarchical_document_candidates,
                "section_candidates": settings.hierarchical_section_candidates,
                "chunk_candidate_multiplier": settings.hierarchical_chunk_candidate_multiplier,
                "max_chunk_candidates": settings.hierarchical_max_chunk_candidates,
                "answer_evidence_level": "source_chunk",
            },
        ),
        implementation=hierarchical_version_aware_retriever,
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
