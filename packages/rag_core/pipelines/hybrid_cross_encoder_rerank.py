from __future__ import annotations

from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL, QueryClassifier
from packages.rag_core.agents.query_graph.tracing import answer_summary, question_summary
from packages.rag_core.agents.shared.retrieval.tracing import evidence_context_summary, evidence_context_trace_metadata, evidence_summary, rerank_input_summary
from packages.rag_core.agents.runtime import GraphRunner, NodeSpec
from packages.rag_core.agents.query_graph.nodes import GenerateAnswerNode
from packages.rag_core.agents.shared.retrieval.nodes import PrepareEvidenceContextNode, RerankNode, RetrieveNode
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.pipelines.baseline import BASELINE_LLM_TOOL, BASELINE_RETRIEVER_TOOL
from packages.rag_core.pipelines.hybrid import HYBRID_KEYWORD_RETRIEVER_TOOL, HYBRID_RETRIEVER_TOOL
from packages.rag_core.agents.query_graph.graph import build_query_classification_node
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.rerankers import Reranker
from packages.rag_core.retrieval.retrievers import Retriever

HYBRID_CROSS_ENCODER_RERANK_RAG_NAME = "hybrid_cross_encoder_rerank_rag"
HYBRID_CROSS_ENCODER_RERANK_RAG_VERSION = "0.4.0"
HYBRID_CROSS_ENCODER_RERANKER_TOOL = "reranker.cross_encoder"
HYBRID_CROSS_ENCODER_RERANK_RAG_CONFIG = PipelineConfig(
    name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    version=HYBRID_CROSS_ENCODER_RERANK_RAG_VERSION,
    description="Classify the query, run hybrid retrieval, apply a local cross-encoder reranker, validate metadata scope, and generate an answer.",
    tool_names=(
        QUERY_CLASSIFIER_TOOL,
        BASELINE_RETRIEVER_TOOL,
        HYBRID_KEYWORD_RETRIEVER_TOOL,
        HYBRID_RETRIEVER_TOOL,
        HYBRID_CROSS_ENCODER_RERANKER_TOOL,
        BASELINE_LLM_TOOL,
    ),
    metadata={
        "stages": ("classify_query", "retrieve", "rerank", "prepare_evidence_context", "generate_answer"),
        "retrieval_strategy": "hybrid",
        "fusion_method": "weighted_reciprocal_rank_fusion",
        "reranking_strategy": "cross_encoder_pairwise_relevance",
    },
)


def build_hybrid_cross_encoder_rerank_rag_graph(
    *,
    retriever: Retriever,
    reranker: Reranker,
    llm_provider: LLMProvider,
    candidate_multiplier: int,
    max_candidates: int,
    query_classifier: QueryClassifier | None = None,
) -> GraphRunner:
    """Build the classified hybrid retrieval, reranking, and generation graph."""

    return GraphRunner(
        name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        version=HYBRID_CROSS_ENCODER_RERANK_RAG_VERSION,
        nodes=[
            build_query_classification_node(query_classifier),
            NodeSpec(
                node=RetrieveNode(
                    retriever,
                    candidate_multiplier=candidate_multiplier,
                    max_candidates=max_candidates,
                ),
                input_summary=question_summary,
                output_summary=evidence_summary,
            ),
            NodeSpec(
                node=RerankNode(reranker),
                input_summary=rerank_input_summary,
                output_summary=evidence_summary,
            ),
            NodeSpec(
                node=PrepareEvidenceContextNode(),
                input_summary=evidence_summary,
                output_summary=evidence_context_summary,
                trace_metadata=evidence_context_trace_metadata,
            ),
            NodeSpec(
                node=GenerateAnswerNode(llm_provider),
                input_summary=evidence_context_summary,
                output_summary=answer_summary,
            ),
        ],
    )
