from __future__ import annotations

from typing import cast

from packages.indexer_bootstrap.config import Settings
from packages.rag_core.agents.shared.retrieval import RetrievalPlanExecution
from packages.rag_core.agents.tools import ToolRegistry
from packages.rag_core.pipelines import (
    AGENTIC_RAG_CONFIG,
    BASELINE_LLM_TOOL,
    BASELINE_RAG_NAME,
    BASELINE_RAG_VERSION,
    BASELINE_RETRIEVER_TOOL,
    CONTEXTUAL_RAG_NAME,
    CONTEXTUAL_RAG_VERSION,
    CONTEXTUAL_RETRIEVER_TOOL,
    HIERARCHICAL_RAG_NAME,
    HIERARCHICAL_RAG_VERSION,
    HIERARCHICAL_RETRIEVER_TOOL,
    HYBRID_CROSS_ENCODER_RERANKER_TOOL,
    HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    HYBRID_CROSS_ENCODER_RERANK_RAG_VERSION,
    HYBRID_RAG_NAME,
    HYBRID_RAG_VERSION,
    HYBRID_RETRIEVER_TOOL,
    MULTI_QUERY_RAG_NAME,
    MULTI_QUERY_RAG_VERSION,
    MULTI_QUERY_RETRIEVER_TOOL,
    PipelineRegistry,
    build_agentic_rag_graph,
)
from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL, QueryClassifier
from packages.rag_core.query_understanding.decomposition import (
    INFORMATION_NEED_DECOMPOSER_TOOL,
    InformationNeedDecomposer,
)
from packages.rag_core.query_understanding.planning import (
    RETRIEVAL_PLANNER_TOOL,
    InformationNeedRetrievalPlanner,
    RetrievalStrategy,
)
from packages.rag_core.retrieval.arbitration import EVIDENCE_ARBITRATOR_TOOL, EvidenceArbitrator
from packages.rag_core.retrieval.document_selection import (
    DOCUMENT_CANDIDATE_SELECTOR_TOOL,
    PRIMARY_DOCUMENT_DETECTOR_TOOL,
    DocumentCandidateSelector,
    PrimaryDocumentDetector,
)
from packages.rag_core.retrieval.graders import EVIDENCE_GRADER_TOOL, EvidenceGrader
from packages.rag_core.retrieval.retry import RETRIEVAL_RETRY_POLICY_TOOL, RetrievalRetryPolicy
from packages.rag_core.retrieval.rerankers import Reranker
from packages.rag_core.retrieval.retrievers import Retriever


def build_agentic_retrieval_executions(
    *,
    settings: Settings,
    tools: ToolRegistry,
) -> dict[str, RetrievalPlanExecution]:
    """Resolve the fixed retrieval capabilities available to the agentic graph."""

    return {
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
        HIERARCHICAL_RAG_NAME: RetrievalPlanExecution(
            pipeline_name=HIERARCHICAL_RAG_NAME,
            pipeline_version=HIERARCHICAL_RAG_VERSION,
            strategy=RetrievalStrategy.HIERARCHICAL,
            retriever=cast(Retriever, tools.resolve(HIERARCHICAL_RETRIEVER_TOOL)),
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
    }


def register_agentic_query_pipeline(
    registry: PipelineRegistry,
    *,
    settings: Settings,
    tools: ToolRegistry,
) -> None:
    """Register the adaptive query graph separately from fixed pipelines."""

    registry.register(
        config=AGENTIC_RAG_CONFIG,
        factory=lambda: build_agentic_rag_graph(
            query_classifier=cast(QueryClassifier, tools.resolve(QUERY_CLASSIFIER_TOOL)),
            information_need_decomposer=cast(
                InformationNeedDecomposer,
                tools.resolve(INFORMATION_NEED_DECOMPOSER_TOOL),
            ),
            retrieval_planner=cast(
                InformationNeedRetrievalPlanner,
                tools.resolve(RETRIEVAL_PLANNER_TOOL),
            ),
            executions=build_agentic_retrieval_executions(settings=settings, tools=tools),
            evidence_grader=cast(EvidenceGrader, tools.resolve(EVIDENCE_GRADER_TOOL)),
            retry_policy=cast(RetrievalRetryPolicy, tools.resolve(RETRIEVAL_RETRY_POLICY_TOOL)),
            llm_provider=cast(LLMProvider, tools.resolve(BASELINE_LLM_TOOL)),
            evidence_arbitrator=cast(EvidenceArbitrator, tools.resolve(EVIDENCE_ARBITRATOR_TOOL)),
            primary_document_detector=cast(
                PrimaryDocumentDetector,
                tools.resolve(PRIMARY_DOCUMENT_DETECTOR_TOOL),
            ),
            document_candidate_selector=cast(
                DocumentCandidateSelector,
                tools.resolve(DOCUMENT_CANDIDATE_SELECTOR_TOOL),
            ),
            max_retries_per_information_need=settings.retrieval_retry_max_retries,
            max_total_retrieval_attempts=settings.retrieval_retry_max_total_attempts,
            max_accumulated_evidence=settings.retrieval_retry_max_accumulated_evidence,
            max_reclassifications_per_information_need=settings.retrieval_retry_max_reclassifications,
        ),
    )
