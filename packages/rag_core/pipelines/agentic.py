from __future__ import annotations

from collections.abc import Mapping

from packages.rag_core.agents import END, ConditionalEdge, ConditionalGraphRunner, GraphRunner, NodeSpec
from packages.rag_core.agents.graph import (
    answer_summary,
    information_need_decomposition_summary,
    information_need_decomposition_trace_metadata,
    information_need_resolution_summary,
    information_need_resolution_trace_metadata,
)
from packages.rag_core.agents.nodes import (
    AggregateInformationNeedsNode,
    ClassifyInformationNeedNode,
    CompleteInformationNeedNode,
    DecideInformationNeedNode,
    DecomposeInformationNeedsNode,
    ExecuteInformationNeedPlanNode,
    ExecuteRetrievalPlanNode,
    GenerateAnswerNode,
    GradeInformationNeedNode,
    InitializeInformationNeedWorkNode,
    PlanInformationNeedNode,
    ResolveInformationNeedsNode,
    RetrievalPlanExecution,
    SelectInformationNeedNode,
)
from packages.rag_core.agents.state import QueryState
from packages.rag_core.agents.work_items import InformationNeedRoute
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
from packages.rag_core.query_understanding.planning import (
    RETRIEVAL_PLANNER_TOOL,
    InformationNeedRetrievalPlanner,
)
from packages.rag_core.retrieval.graders import EVIDENCE_GRADER_TOOL, EvidenceGrader
from packages.rag_core.retrieval.retry import RETRIEVAL_RETRY_POLICY_TOOL, RetrievalRetryPolicy

AGENTIC_RAG_NAME = "agentic_rag"
AGENTIC_RAG_VERSION = "0.8.0"
INFORMATION_NEED_GRAPH_NAME = "information_need_resolution"
INFORMATION_NEED_GRAPH_VERSION = "0.1.0"

AGENTIC_RAG_CONFIG = PipelineConfig(
    name=AGENTIC_RAG_NAME,
    version=AGENTIC_RAG_VERSION,
    description=(
        "Classify and decompose the query, then resolve every information need through a reusable bounded subgraph "
        "that independently classifies, plans, retrieves, grades, and retries that item before aggregating complete "
        "or explicitly partial answer evidence."
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
        "graph_mode": "hierarchical_top_level_with_cyclic_information_need_subgraph",
        "top_level_stages": (
            "classify_query",
            "decompose_information_needs",
            "initialize_information_need_work",
            "resolve_information_needs",
            "aggregate_information_needs",
            "generate_answer",
        ),
        "information_need_subgraph_stages": (
            "select_information_need",
            "classify_information_need",
            "plan_information_need",
            "execute_information_need_plan",
            "grade_information_need",
            "decide_information_need",
            "complete_information_need",
        ),
        "information_need_routes": (
            "supported_to_complete",
            "insufficient_to_plan",
            "low_confidence_missing_to_reclassify",
            "exhausted_to_complete",
            "queue_empty_to_parent",
        ),
        "selection_mode": "per_information_need_classification_and_planning",
        "selectable_strategies": ("baseline", "hybrid", "contextual", "multi_query", "rerank"),
        "retry_mode": "per_information_need_bounded_cycles",
        "retry_budget_mode": "per_information_need_and_query_global_limits",
        "answer_evidence_mode": "union_of_per_information_need_grader_approved_evidence",
        "partial_answer_mode": "explicit_unresolved_information_need_disclosure",
    },
)


def build_agentic_rag_graph(
    *,
    query_classifier: QueryClassifier,
    information_need_decomposer: InformationNeedDecomposer,
    retrieval_planner: InformationNeedRetrievalPlanner,
    executions: Mapping[str, RetrievalPlanExecution],
    evidence_grader: EvidenceGrader,
    retry_policy: RetrievalRetryPolicy,
    llm_provider: LLMProvider,
    max_retries_per_information_need: int = 2,
    max_total_retrieval_attempts: int = 20,
    max_accumulated_evidence: int = 40,
    max_reclassifications_per_information_need: int = 1,
) -> GraphRunner:
    """Build the hierarchical agent graph and reusable cyclic subgraph."""

    if max_retries_per_information_need < 0:
        raise ValueError("max_retries_per_information_need must not be negative.")
    max_attempts_per_need = max_retries_per_information_need + 1
    retrieval_node = ExecuteRetrievalPlanNode(executions)

    information_need_subgraph = ConditionalGraphRunner(
        name=INFORMATION_NEED_GRAPH_NAME,
        version=INFORMATION_NEED_GRAPH_VERSION,
        entry_point="select_information_need",
        max_steps=max(32, max_total_retrieval_attempts * 7 + 16),
        nodes={
            "select_information_need": NodeSpec(
                node=SelectInformationNeedNode(),
                output_summary=_active_need_summary,
            ),
            "classify_information_need": NodeSpec(
                node=ClassifyInformationNeedNode(query_classifier),
                input_summary=_active_need_summary,
                output_summary=_active_need_classification_summary,
                trace_metadata=_active_need_metadata,
            ),
            "plan_information_need": NodeSpec(
                node=PlanInformationNeedNode(
                    retrieval_planner,
                    available_pipeline_names=retrieval_node.available_pipeline_names,
                    max_total_attempts=max_total_retrieval_attempts,
                ),
                input_summary=_active_need_grade_summary,
                output_summary=_active_need_plan_summary,
                trace_metadata=_active_need_metadata,
            ),
            "execute_information_need_plan": NodeSpec(
                node=ExecuteInformationNeedPlanNode(
                    retrieval_node,
                    max_total_attempts=max_total_retrieval_attempts,
                    max_accumulated_evidence=max_accumulated_evidence,
                ),
                input_summary=_active_need_plan_summary,
                output_summary=_active_need_lookup_summary,
                trace_metadata=_active_need_metadata,
            ),
            "grade_information_need": NodeSpec(
                node=GradeInformationNeedNode(evidence_grader),
                input_summary=_active_need_lookup_summary,
                output_summary=_active_need_grade_summary,
                trace_metadata=_active_need_metadata,
            ),
            "decide_information_need": NodeSpec(
                node=DecideInformationNeedNode(
                    retry_policy,
                    max_total_attempts=max_total_retrieval_attempts,
                    max_reclassifications=max_reclassifications_per_information_need,
                ),
                input_summary=_active_need_grade_summary,
                output_summary=_active_need_decision_summary,
                trace_metadata=_active_need_metadata,
            ),
            "complete_information_need": NodeSpec(
                node=CompleteInformationNeedNode(),
                input_summary=_active_need_decision_summary,
                output_summary=_completed_need_summary,
                trace_metadata=_completed_need_metadata,
            ),
        },
        edges={
            "select_information_need": ConditionalEdge(
                resolver=lambda state: "done" if state.active_information_need_id is None else "active",
                routes={"active": "classify_information_need", "done": END},
            ),
            "classify_information_need": "plan_information_need",
            "plan_information_need": ConditionalEdge(
                resolver=_route_after_information_need_planning,
                routes={"execute": "execute_information_need_plan", "complete": "complete_information_need"},
            ),
            "execute_information_need_plan": "grade_information_need",
            "grade_information_need": "decide_information_need",
            "decide_information_need": ConditionalEdge(
                resolver=lambda state: _active_route(state).value,
                routes={
                    InformationNeedRoute.RETRY.value: "plan_information_need",
                    InformationNeedRoute.RECLASSIFY.value: "classify_information_need",
                    InformationNeedRoute.COMPLETE_SUPPORTED.value: "complete_information_need",
                    InformationNeedRoute.COMPLETE_EXHAUSTED.value: "complete_information_need",
                },
            ),
            "complete_information_need": "select_information_need",
        },
    )

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
                node=InitializeInformationNeedWorkNode(
                    max_attempts_per_information_need=max_attempts_per_need,
                    max_total_attempts=max_total_retrieval_attempts,
                ),
                input_summary=information_need_decomposition_summary,
                output_summary=lambda state: (
                    f"queued_information_needs={len(state.pending_information_need_ids)}; "
                    f"max_attempts_per_need={max_attempts_per_need}; "
                    f"max_total_attempts={max_total_retrieval_attempts}"
                ),
            ),
            NodeSpec(
                node=ResolveInformationNeedsNode(information_need_subgraph),
                input_summary=lambda state: f"queued_information_needs={len(state.pending_information_need_ids)}",
                output_summary=lambda state: (
                    f"processed_information_needs={len(state.information_need_executions)}; "
                    f"retrieval_attempts={state.total_information_need_retrieval_attempts}"
                ),
            ),
            NodeSpec(
                node=AggregateInformationNeedsNode(
                    subgraph_name=INFORMATION_NEED_GRAPH_NAME,
                    max_total_attempts=max_total_retrieval_attempts,
                    max_attempts_per_information_need=max_attempts_per_need,
                ),
                input_summary=lambda state: f"evidence_count={len(state.retrieved_evidence)}",
                output_summary=information_need_resolution_summary,
                trace_metadata=information_need_resolution_trace_metadata,
            ),
            NodeSpec(
                node=GenerateAnswerNode(llm_provider),
                input_summary=information_need_resolution_summary,
                output_summary=answer_summary,
            ),
        ],
    )


def _route_after_information_need_planning(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None:
        raise RuntimeError("Information-need planning routing requires an active work item.")
    return "execute" if execution.current_plan is not None else "complete"


def _active_route(state: QueryState) -> InformationNeedRoute:
    execution = state.active_information_need_execution
    if execution is None or execution.next_route is None:
        raise RuntimeError("The active information need has no graph route decision.")
    return execution.next_route


def _active_need_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None:
        return f"active_information_need=none; pending={len(state.pending_information_need_ids)}"
    return (
        f"information_need_id={execution.information_need.need_id}; "
        f"description={execution.information_need.description!r}; attempts={execution.attempts_used}/{execution.max_attempts}"
    )


def _active_need_classification_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None or execution.classification is None:
        return "information_need_classification=missing"
    return (
        f"information_need_id={execution.information_need.need_id}; "
        f"query_type={execution.classification.query_type.value}; "
        f"confidence={execution.classification.confidence:.2f}; "
        f"classification_count={len(execution.classification_history)}"
    )


def _active_need_plan_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None:
        return "information_need_plan=missing"
    if execution.current_plan is None:
        return f"information_need_id={execution.information_need.need_id}; stop_reason={execution.stop_reason}"
    plan = execution.current_plan
    return (
        f"information_need_id={plan.information_need_id}; attempt={plan.attempt_number}; "
        f"strategy={plan.strategy.value}; pipeline={plan.selected_pipeline_name}; top_k={plan.top_k}; "
        f"query={plan.query!r}"
    )


def _active_need_lookup_summary(state: QueryState) -> str:
    lookup = state.metadata.get("active_information_need_lookup")
    if not isinstance(lookup, dict):
        return "information_need_lookup=missing"
    return (
        f"information_need_id={lookup.get('information_need_id')}; attempt={lookup.get('attempt_number')}; "
        f"retrieved={lookup.get('retrieved_count')}; unique_added={lookup.get('unique_evidence_added')}"
    )


def _active_need_grade_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None or execution.final_grade is None:
        return "information_need_grade=pending"
    return (
        f"information_need_id={execution.information_need.need_id}; status={execution.final_grade.status.value}; "
        f"coverage={execution.final_grade.coverage_score:.2f}; attempts={execution.attempts_used}/{execution.max_attempts}"
    )


def _active_need_decision_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None or execution.next_route is None:
        return "information_need_decision=missing"
    return (
        f"information_need_id={execution.information_need.need_id}; route={execution.next_route.value}; "
        f"stop_reason={execution.stop_reason or 'none'}"
    )


def _active_need_metadata(state: QueryState) -> dict[str, object]:
    execution = state.active_information_need_execution
    return {"information_need_execution": execution.to_metadata()} if execution is not None else {}


def _completed_need_summary(state: QueryState) -> str:
    completed = state.metadata.get("last_completed_information_need")
    if not isinstance(completed, dict):
        return f"completed_information_need=missing; pending={len(state.pending_information_need_ids)}"
    return (
        f"information_need_id={completed.get('information_need_id')}; "
        f"status={completed.get('status')}; attempts={completed.get('attempts_used')}; "
        f"pending={len(state.pending_information_need_ids)}"
    )


def _completed_need_metadata(state: QueryState) -> dict[str, object]:
    completed = state.metadata.get("last_completed_information_need")
    return {"information_need_execution": completed} if isinstance(completed, dict) else {}
