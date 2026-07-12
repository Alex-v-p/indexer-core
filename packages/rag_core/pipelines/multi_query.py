from __future__ import annotations

from packages.rag_core.agents.graph import GraphRunner, NodeSpec, answer_summary, evidence_summary, question_summary
from packages.rag_core.agents.state import QueryState
from packages.rag_core.agents.nodes import GenerateAnswerNode, RetrieveNode
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.pipelines.baseline import BASELINE_LLM_TOOL, BASELINE_RETRIEVER_TOOL
from packages.rag_core.pipelines.hybrid import HYBRID_KEYWORD_RETRIEVER_TOOL, HYBRID_RETRIEVER_TOOL
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.retrievers import Retriever

MULTI_QUERY_RAG_NAME = "multi_query_rag"
MULTI_QUERY_RAG_VERSION = "0.1.0"
MULTI_QUERY_GENERATOR_TOOL = "query_generator.llm_variants"
MULTI_QUERY_RETRIEVER_TOOL = "retriever.multi_query_rrf"
MULTI_QUERY_RAG_CONFIG = PipelineConfig(
    name=MULTI_QUERY_RAG_NAME,
    version=MULTI_QUERY_RAG_VERSION,
    description=(
        "LLM query expansion followed by per-variant hybrid retrieval and weighted reciprocal-rank fusion."
    ),
    tool_names=(
        BASELINE_RETRIEVER_TOOL,
        HYBRID_KEYWORD_RETRIEVER_TOOL,
        HYBRID_RETRIEVER_TOOL,
        MULTI_QUERY_GENERATOR_TOOL,
        MULTI_QUERY_RETRIEVER_TOOL,
        BASELINE_LLM_TOOL,
    ),
    metadata={
        "stages": ("retrieve", "generate_answer"),
        "internal_retrieval_stages": ("generate_query_variants", "retrieve_each", "fuse"),
        "retrieval_strategy": "multi_query_hybrid",
        "base_retrieval_strategy": "hybrid",
        "fusion_method": "weighted_reciprocal_rank_fusion",
    },
)


def build_multi_query_rag_graph(*, retriever: Retriever, llm_provider: LLMProvider) -> GraphRunner:
    """Build the multi-query graph: expand/retrieve/fuse → generate answer."""

    return GraphRunner(
        name=MULTI_QUERY_RAG_NAME,
        version=MULTI_QUERY_RAG_VERSION,
        nodes=[
            NodeSpec(
                node=RetrieveNode(retriever),
                input_summary=question_summary,
                output_summary=multi_query_evidence_summary,
            ),
            NodeSpec(
                node=GenerateAnswerNode(llm_provider),
                input_summary=evidence_summary,
                output_summary=answer_summary,
            ),
        ],
    )


def multi_query_evidence_summary(state: QueryState) -> str:
    retrieval = state.metadata.get("retrieval", {})
    return (
        f"evidence_count={len(state.retrieved_evidence)}; "
        f"query_count={retrieval.get('query_count', 0)}; "
        f"generated_variants={retrieval.get('generated_variant_count', 0)}; "
        f"generation_fallback={retrieval.get('generation_fallback_used', False)}"
    )
