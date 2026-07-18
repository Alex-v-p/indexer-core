from __future__ import annotations

from collections.abc import Mapping

from packages.rag_core.agents.graph import (
    GraphRunner,
    NodeSpec,
    answer_summary,
    evidence_grading_input_summary,
    evidence_grading_summary,
    evidence_grading_trace_metadata,
    information_need_decomposition_summary,
    information_need_decomposition_trace_metadata,
    planned_retrieval_summary,
    planned_retrieval_trace_metadata,
    retrieval_plan_summary,
    retrieval_plan_trace_metadata,
    retrieval_planning_input_summary,
    retrieval_retry_input_summary,
    retrieval_retry_summary,
    retrieval_retry_trace_metadata,
)
from packages.rag_core.agents.nodes import (
    DecomposeInformationNeedsNode,
    ExecuteRetrievalPlanNode,
    GenerateAnswerNode,
    GradeEvidenceNode,
    PlanRetrievalNode,
    RetrievalPlanExecution,
    RetryRetrievalNode,
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
from packages.rag_core.query_understanding.decomposition import (
    INFORMATION_NEED_DECOMPOSER_TOOL,
    InformationNeedDecomposer,
)
from packages.rag_core.query_understanding.planning import RETRIEVAL_PLANNER_TOOL, RetrievalPlanner
from packages.rag_core.retrieval.graders import EVIDENCE_GRADER_TOOL, EvidenceGrader
from packages.rag_core.retrieval.retry import RETRIEVAL_RETRY_POLICY_TOOL, RetrievalRetryPolicy

AGENTIC_RAG_NAME = "agentic_rag"
AGENTIC_RAG_VERSION = "0.5.0"
AGENTIC_RAG_CONFIG = PipelineConfig(
    name=AGENTIC_RAG_NAME,
    version=AGENTIC_RAG_VERSION,
    description=(
        "Classify the query, independently decompose its answer requirements, plan and execute the most suitable "
        "registered retrieval strategy, grade every retrieved chunk and information need, apply bounded query/top-k/"
        "pipeline fallbacks when evidence is weak, and generate an answer only when all required needs are supported."
    ),
    tool_names=(
        QUERY_CLASSIFIER_TOOL,
        INFORMATION_NEED_DECOMPOSER_TOOL,
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
        EVIDENCE_GRADER_TOOL,
        RETRIEVAL_RETRY_POLICY_TOOL,
        BASELINE_LLM_TOOL,
    ),
    metadata={
        "stages": (
            "classify_query",
            "decompose_information_needs",
            "plan_retrieval",
            "execute_retrieval_plan",
            "grade_evidence",
            "retry_retrieval",
            "generate_answer",
        ),
        "selection_mode": "classification_and_decomposition_driven",
        "selectable_strategies": ("baseline", "hybrid", "contextual", "multi_query", "rerank"),
        "evidence_gate": "block_generation_until_all_required_information_needs_are_supported",
        "retry_mode": "bounded_query_top_k_and_pipeline_escalation",
    },
)


def build_agentic_rag_graph(
    *,
    query_classifier: QueryClassifier,
    information_need_decomposer: InformationNeedDecomposer,
    retrieval_planner: RetrievalPlanner,
    executions: Mapping[str, RetrievalPlanExecution],
    evidence_grader: EvidenceGrader,
    retry_policy: RetrievalRetryPolicy,
    llm_provider: LLMProvider,
) -> GraphRunner:
    """Build the agentic graph with bounded post-grading retrieval fallbacks."""

    retrieval_node = ExecuteRetrievalPlanNode(executions)
    grading_node = GradeEvidenceNode(evidence_grader)

    return GraphRunner(
        name=AGENTIC_RAG_NAME,
        version=AGENTIC_RAG_VERSION,
        nodes=[
            build_query_classification_node(query_classifier),
            NodeSpec(
                node=DecomposeInformationNeedsNode(information_need_decomposer),
                input_summary=lambda state: f"question={state.question!r}",
                output_summary=information_need_decomposition_summary,
                trace_metadata=information_need_decomposition_trace_metadata,
            ),
            NodeSpec(
                node=PlanRetrievalNode(retrieval_planner),
                input_summary=retrieval_planning_input_summary,
                output_summary=retrieval_plan_summary,
                trace_metadata=retrieval_plan_trace_metadata,
            ),
            NodeSpec(
                node=retrieval_node,
                input_summary=retrieval_plan_summary,
                output_summary=planned_retrieval_summary,
                trace_metadata=planned_retrieval_trace_metadata,
            ),
            NodeSpec(
                node=grading_node,
                input_summary=evidence_grading_input_summary,
                output_summary=evidence_grading_summary,
                trace_metadata=evidence_grading_trace_metadata,
            ),
            NodeSpec(
                node=RetryRetrievalNode(
                    retry_policy=retry_policy,
                    retrieval_node=retrieval_node,
                    grading_node=grading_node,
                ),
                input_summary=retrieval_retry_input_summary,
                output_summary=retrieval_retry_summary,
                trace_metadata=retrieval_retry_trace_metadata,
            ),
            NodeSpec(
                node=GenerateAnswerNode(llm_provider),
                input_summary=evidence_grading_summary,
                output_summary=answer_summary,
            ),
        ],
    )
