from __future__ import annotations

from packages.rag_core.agents.graph import (
    GraphRunner,
    NodeSpec,
    answer_summary,
    evidence_summary,
    question_summary,
    rerank_input_summary,
)
from packages.rag_core.agents.nodes import GenerateAnswerNode, RerankNode, RetrieveNode
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.pipelines.baseline import BASELINE_LLM_TOOL, BASELINE_RETRIEVER_TOOL
from packages.rag_core.pipelines.hybrid import HYBRID_KEYWORD_RETRIEVER_TOOL, HYBRID_RETRIEVER_TOOL
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.rerankers import Reranker
from packages.rag_core.retrieval.retrievers import Retriever

HYBRID_LLM_RERANK_RAG_NAME = "hybrid_llm_rerank_rag"
HYBRID_LLM_RERANK_RAG_VERSION = "0.2.0"
HYBRID_LLM_RERANKER_TOOL = "reranker.ollama"
HYBRID_LLM_RERANK_RAG_CONFIG = PipelineConfig(
    name=HYBRID_LLM_RERANK_RAG_NAME,
    version=HYBRID_LLM_RERANK_RAG_VERSION,
    description="Hybrid vector/BM25 retrieval followed by resilient query-aware Ollama reranking and answer generation.",
    tool_names=(
        BASELINE_RETRIEVER_TOOL,
        HYBRID_KEYWORD_RETRIEVER_TOOL,
        HYBRID_RETRIEVER_TOOL,
        HYBRID_LLM_RERANKER_TOOL,
        BASELINE_LLM_TOOL,
    ),
    metadata={
        "stages": ("retrieve", "rerank", "generate_answer"),
        "retrieval_strategy": "hybrid",
        "fusion_method": "weighted_reciprocal_rank_fusion",
        "reranking_strategy": "ollama_pointwise_relevance",
        "incomplete_response_policy": "retry_then_preserve_original_rank",
    },
)


def build_hybrid_llm_rerank_rag_graph(
    *,
    retriever: Retriever,
    reranker: Reranker,
    llm_provider: LLMProvider,
    candidate_multiplier: int,
    max_candidates: int,
) -> GraphRunner:
    """Build the hybrid LLM-reranking graph: retrieve candidates → rerank → generate."""

    return GraphRunner(
        name=HYBRID_LLM_RERANK_RAG_NAME,
        version=HYBRID_LLM_RERANK_RAG_VERSION,
        nodes=[
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
                node=GenerateAnswerNode(llm_provider),
                input_summary=evidence_summary,
                output_summary=answer_summary,
            ),
        ],
    )
