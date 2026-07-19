from __future__ import annotations

from packages.rag_core.agents.query_graph.graph import build_query_classification_node
from packages.rag_core.agents.query_graph.nodes import GenerateAnswerNode
from packages.rag_core.agents.query_graph.tracing import answer_summary, question_summary
from packages.rag_core.agents.runtime import GraphRunner, NodeSpec
from packages.rag_core.agents.shared.retrieval.nodes import PrepareEvidenceContextNode, RetrieveNode
from packages.rag_core.agents.shared.retrieval.tracing import (
    evidence_context_summary,
    evidence_context_trace_metadata,
    evidence_summary,
)
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.pipelines.baseline import BASELINE_LLM_TOOL
from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL, QueryClassifier
from packages.rag_core.retrieval.retrievers import Retriever

HIERARCHICAL_RAG_NAME = "hierarchical_rag"
HIERARCHICAL_RAG_VERSION = "0.1.0"
HIERARCHICAL_RETRIEVER_TOOL = "retriever.hierarchical"
HIERARCHICAL_RAG_CONFIG = PipelineConfig(
    name=HIERARCHICAL_RAG_NAME,
    version=HIERARCHICAL_RAG_VERSION,
    description=(
        "Classify the query, route through document and semantic-section summaries, retrieve precise source chunks "
        "from the selected sections, and generate a citation-aware answer."
    ),
    tool_names=(QUERY_CLASSIFIER_TOOL, HIERARCHICAL_RETRIEVER_TOOL, BASELINE_LLM_TOOL),
    metadata={
        "stages": (
            "classify_query",
            "retrieve_document_summaries",
            "retrieve_section_summaries",
            "retrieve_scoped_chunks",
            "prepare_evidence_context",
            "generate_answer",
        ),
        "retrieval_strategy": "hierarchical_document_section_chunk",
        "routing_levels": ("document_summary", "section_summary", "source_chunk"),
        "answer_evidence_level": "source_chunk",
    },
)


def build_hierarchical_rag_graph(
    *,
    retriever: Retriever,
    llm_provider: LLMProvider,
    query_classifier: QueryClassifier | None = None,
) -> GraphRunner:
    """Build classify → hierarchical retrieve → source-grounded answer."""

    return GraphRunner(
        name=HIERARCHICAL_RAG_NAME,
        version=HIERARCHICAL_RAG_VERSION,
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
