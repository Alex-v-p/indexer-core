from __future__ import annotations

from packages.rag_core.agents.graph import GraphRunner, NodeSpec, answer_summary, evidence_summary, question_summary
from packages.rag_core.agents.nodes import GenerateAnswerNode, RetrieveNode
from packages.rag_core.providers import LLMProvider
from packages.rag_core.retrieval.retrievers import Retriever

BASELINE_RAG_NAME = "baseline_rag"
BASELINE_RAG_VERSION = "0.1.0"


def build_baseline_rag_graph(*, retriever: Retriever, llm_provider: LLMProvider) -> GraphRunner:
    """Build the first Phase 1 graph: retrieve → generate_answer."""

    return GraphRunner(
        name=BASELINE_RAG_NAME,
        version=BASELINE_RAG_VERSION,
        nodes=[
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
