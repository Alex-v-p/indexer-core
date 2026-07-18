from __future__ import annotations

from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL, QueryClassifier
from packages.rag_core.agents.query_graph.tracing import answer_summary, question_summary
from packages.rag_core.agents.shared.retrieval.tracing import evidence_summary
from packages.rag_core.agents.runtime import GraphRunner, NodeSpec
from packages.rag_core.agents.query_graph.nodes import GenerateAnswerNode
from packages.rag_core.agents.shared.retrieval.nodes import RetrieveNode
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.pipelines.baseline import BASELINE_LLM_TOOL, BASELINE_RETRIEVER_TOOL
from packages.rag_core.agents.query_graph.graph import build_query_classification_node
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.retrievers import Retriever

HYBRID_RAG_NAME = "hybrid_rag"
HYBRID_RAG_VERSION = "0.2.0"
HYBRID_KEYWORD_RETRIEVER_TOOL = "retriever.keyword_bm25"
HYBRID_RETRIEVER_TOOL = "retriever.hybrid_rrf"
HYBRID_RAG_CONFIG = PipelineConfig(
    name=HYBRID_RAG_NAME,
    version=HYBRID_RAG_VERSION,
    description="Classify the query, fuse dense-vector and BM25 retrieval, then generate an answer.",
    tool_names=(
        QUERY_CLASSIFIER_TOOL,
        BASELINE_RETRIEVER_TOOL,
        HYBRID_KEYWORD_RETRIEVER_TOOL,
        HYBRID_RETRIEVER_TOOL,
        BASELINE_LLM_TOOL,
    ),
    metadata={
        "stages": ("classify_query", "retrieve", "generate_answer"),
        "retrieval_strategy": "hybrid",
        "fusion_method": "weighted_reciprocal_rank_fusion",
    },
)


def build_hybrid_rag_graph(
    *,
    retriever: Retriever,
    llm_provider: LLMProvider,
    query_classifier: QueryClassifier | None = None,
) -> GraphRunner:
    """Build the hybrid graph: classify → hybrid retrieve → generate_answer."""

    return GraphRunner(
        name=HYBRID_RAG_NAME,
        version=HYBRID_RAG_VERSION,
        nodes=[
            build_query_classification_node(query_classifier),
            NodeSpec(
                node=RetrieveNode(retriever),
                input_summary=question_summary,
                output_summary=evidence_summary,
            ),
            NodeSpec(
                node=GenerateAnswerNode(llm_provider),
                input_summary=evidence_summary,
                output_summary=answer_summary,
            ),
        ],
    )
