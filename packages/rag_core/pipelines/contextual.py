from __future__ import annotations

from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL, QueryClassifier
from packages.rag_core.agents.query_graph.tracing import answer_summary, question_summary
from packages.rag_core.agents.shared.retrieval.tracing import evidence_context_summary, evidence_context_trace_metadata, evidence_summary
from packages.rag_core.agents.runtime import GraphRunner, NodeSpec
from packages.rag_core.agents.query_graph.nodes import GenerateAnswerNode
from packages.rag_core.agents.shared.retrieval.nodes import PrepareEvidenceContextNode, RetrieveNode
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.pipelines.baseline import BASELINE_LLM_TOOL
from packages.rag_core.agents.query_graph.graph import build_query_classification_node
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.retrievers import Retriever

CONTEXTUAL_RAG_NAME = "contextual_rag"
CONTEXTUAL_RAG_VERSION = "0.5.0"
CONTEXTUAL_VECTOR_RETRIEVER_TOOL = "retriever.contextual_vector"
CONTEXTUAL_KEYWORD_RETRIEVER_TOOL = "retriever.contextual_keyword_bm25"
CONTEXTUAL_RETRIEVER_TOOL = "retriever.contextual_hybrid_rrf"
CONTEXTUAL_RAG_CONFIG = PipelineConfig(
    name=CONTEXTUAL_RAG_NAME,
    version=CONTEXTUAL_RAG_VERSION,
    description=(
        "Classify the query, retrieve over document-aware contextualized chunk representations, and return "
        "original source text for answer generation and citations."
    ),
    tool_names=(
        QUERY_CLASSIFIER_TOOL,
        CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
        CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
        CONTEXTUAL_RETRIEVER_TOOL,
        BASELINE_LLM_TOOL,
    ),
    metadata={
        "stages": ("classify_query", "retrieve", "prepare_evidence_context", "generate_answer"),
        "retrieval_strategy": "contextual_hybrid",
        "contextualization_strategy": "adjacent_chunk_window",
        "fusion_method": "weighted_reciprocal_rank_fusion",
        "evidence_text": "original_chunk",
    },
)


def build_contextual_rag_graph(
    *,
    retriever: Retriever,
    llm_provider: LLMProvider,
    query_classifier: QueryClassifier | None = None,
) -> GraphRunner:
    """Build the contextual graph: classify → contextual retrieve → generate answer."""

    return GraphRunner(
        name=CONTEXTUAL_RAG_NAME,
        version=CONTEXTUAL_RAG_VERSION,
        nodes=[
            build_query_classification_node(query_classifier),
            NodeSpec(
                node=RetrieveNode(retriever),
                input_summary=question_summary,
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
