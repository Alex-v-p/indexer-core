from __future__ import annotations

from collections.abc import Mapping

from packages.rag_core.agents.graph import (
    GraphRunner,
    NodeSpec,
    answer_summary,
    classification_summary,
    planned_retrieval_summary,
    planned_retrieval_trace_metadata,
    retrieval_plan_summary,
    retrieval_plan_trace_metadata,
)
from packages.rag_core.agents.nodes import (
    ExecuteRetrievalPlanNode,
    GenerateAnswerNode,
    PlanRetrievalNode,
    RetrievalPlanExecution,
)
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.pipelines.baseline import BASELINE_LLM_TOOL, BASELINE_RETRIEVER_TOOL
from packages.rag_core.pipelines.contextual import (
    CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
    CONTEXTUAL_RETRIEVER_TOOL,
    CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
)
from packages.rag_core.pipelines.hybrid import HYBRID_KEYWORD_RETRIEVER_TOOL, HYBRID_RETRIEVER_TOOL
from packages.rag_core.pipelines.hybrid_cross_encoder_rerank import HYBRID_CROSS_ENCODER_RERANKER_TOOL
from packages.rag_core.pipelines.multi_query import MULTI_QUERY_GENERATOR_TOOL, MULTI_QUERY_RETRIEVER_TOOL
from packages.rag_core.pipelines.query_classification import build_query_classification_node
from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.classification import QUERY_CLASSIFIER_TOOL, QueryClassifier
from packages.rag_core.query_understanding.planning import RETRIEVAL_PLANNER_TOOL, RetrievalPlanner

AGENTIC_RAG_NAME = "agentic_rag"
AGENTIC_RAG_VERSION = "0.1.0"
AGENTIC_RAG_CONFIG = PipelineConfig(
    name=AGENTIC_RAG_NAME,
    version=AGENTIC_RAG_VERSION,
    description=(
        "Classify the query, plan the most suitable registered retrieval strategy, execute it dynamically, "
        "and generate one citation-aware answer."
    ),
    tool_names=(
        QUERY_CLASSIFIER_TOOL,
        RETRIEVAL_PLANNER_TOOL,
        BASELINE_RETRIEVER_TOOL,
        HYBRID_KEYWORD_RETRIEVER_TOOL,
        HYBRID_RETRIEVER_TOOL,
        CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
        CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
        CONTEXTUAL_RETRIEVER_TOOL,
        MULTI_QUERY_GENERATOR_TOOL,
        MULTI_QUERY_RETRIEVER_TOOL,
        HYBRID_CROSS_ENCODER_RERANKER_TOOL,
        BASELINE_LLM_TOOL,
    ),
    metadata={
        "stages": ("classify_query", "plan_retrieval", "execute_retrieval_plan", "generate_answer"),
        "selection_mode": "classification_driven",
        "selectable_strategies": ("baseline", "hybrid", "contextual", "multi_query", "rerank"),
    },
)


def build_agentic_rag_graph(
    *,
    query_classifier: QueryClassifier,
    retrieval_planner: RetrievalPlanner,
    executions: Mapping[str, RetrievalPlanExecution],
    llm_provider: LLMProvider,
) -> GraphRunner:
    """Build the agentic graph: classify → plan → dynamic retrieval → answer."""

    return GraphRunner(
        name=AGENTIC_RAG_NAME,
        version=AGENTIC_RAG_VERSION,
        nodes=[
            build_query_classification_node(query_classifier),
            NodeSpec(
                node=PlanRetrievalNode(retrieval_planner),
                input_summary=classification_summary,
                output_summary=retrieval_plan_summary,
                trace_metadata=retrieval_plan_trace_metadata,
            ),
            NodeSpec(
                node=ExecuteRetrievalPlanNode(executions),
                input_summary=retrieval_plan_summary,
                output_summary=planned_retrieval_summary,
                trace_metadata=planned_retrieval_trace_metadata,
            ),
            NodeSpec(
                node=GenerateAnswerNode(llm_provider),
                input_summary=planned_retrieval_summary,
                output_summary=answer_summary,
            ),
        ],
    )
