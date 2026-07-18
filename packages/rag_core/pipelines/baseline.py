from __future__ import annotations

from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL, QueryClassifier
from packages.rag_core.agents.query_graph.tracing import answer_summary, question_summary
from packages.rag_core.agents.retrieval.tracing import evidence_summary
from packages.rag_core.agents.runtime import GraphRunner, NodeSpec
from packages.rag_core.agents.query_graph.nodes import GenerateAnswerNode
from packages.rag_core.agents.retrieval.nodes import RetrieveNode
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.agents.query_graph.graph import build_query_classification_node
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.retrievers import Retriever

BASELINE_RAG_NAME = "baseline_rag"
BASELINE_RAG_VERSION = "0.3.0"
BASELINE_RETRIEVER_TOOL = "retriever.vector"
BASELINE_LLM_TOOL = "generator.answer"
BASELINE_RAG_CONFIG = PipelineConfig(
    name=BASELINE_RAG_NAME,
    version=BASELINE_RAG_VERSION,
    description="Classify the query, run dense-vector retrieval, then generate a citation-aware answer.",
    tool_names=(QUERY_CLASSIFIER_TOOL, BASELINE_RETRIEVER_TOOL, BASELINE_LLM_TOOL),
    metadata={"stages": ("classify_query", "retrieve", "generate_answer")},
)


def build_baseline_rag_graph(
    *,
    retriever: Retriever,
    llm_provider: LLMProvider,
    query_classifier: QueryClassifier | None = None,
) -> GraphRunner:
    """Build the baseline graph: classify → retrieve → generate_answer."""

    return GraphRunner(
        name=BASELINE_RAG_NAME,
        version=BASELINE_RAG_VERSION,
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
