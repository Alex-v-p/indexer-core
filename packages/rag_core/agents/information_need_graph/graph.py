from __future__ import annotations

from packages.rag_core.agents.information_need_graph.nodes import (
    ClassifyInformationNeedNode,
    CompleteInformationNeedNode,
    DecideInformationNeedNode,
    DetectPrimaryDocumentNode,
    ExecuteInformationNeedPlanNode,
    GradeInformationNeedNode,
    PlanInformationNeedNode,
    SelectInformationNeedNode,
    ValidateInformationNeedConstraintsNode,
)
from packages.rag_core.agents.information_need_graph.routes import InformationNeedRoute
from packages.rag_core.agents.information_need_graph.tracing import (
    active_information_need_trace_metadata,
    active_need_classification_summary,
    active_need_constraint_validation_summary,
    active_need_decision_summary,
    active_need_grade_summary,
    active_need_lookup_metadata,
    active_need_lookup_summary,
    active_need_metadata,
    active_need_plan_summary,
    active_need_summary,
    completed_need_metadata,
    completed_need_summary,
    primary_document_detection_metadata,
    primary_document_detection_summary,
)
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.shared.retrieval import RetrievalPlanExecutor
from packages.rag_core.agents.runtime import END, ConditionalEdge, ConditionalGraphRunner, NodeSpec
from packages.rag_core.query_understanding.classification import QueryClassifier
from packages.rag_core.query_understanding.planning import InformationNeedRetrievalPlanner
from packages.rag_core.retrieval.graders import EvidenceGrader
from packages.rag_core.retrieval.document_selection import PrimaryDocumentDetector
from packages.rag_core.retrieval.retry import RetrievalRetryPolicy

INFORMATION_NEED_GRAPH_NAME = "information_need_resolution"
INFORMATION_NEED_GRAPH_VERSION = "0.3.0"


def build_information_need_graph(
    *,
    query_classifier: QueryClassifier,
    retrieval_planner: InformationNeedRetrievalPlanner,
    retrieval_executor: RetrievalPlanExecutor,
    evidence_grader: EvidenceGrader,
    primary_document_detector: PrimaryDocumentDetector,
    retry_policy: RetrievalRetryPolicy,
    max_total_retrieval_attempts: int,
    max_accumulated_evidence: int,
    max_reclassifications_per_information_need: int,
) -> ConditionalGraphRunner:
    """Build the bounded cyclic graph that resolves queued information needs."""

    return ConditionalGraphRunner(
        name=INFORMATION_NEED_GRAPH_NAME,
        version=INFORMATION_NEED_GRAPH_VERSION,
        entry_point="select_information_need",
        max_steps=max(32, max_total_retrieval_attempts * 8 + 16),
        trace_metadata=active_information_need_trace_metadata,
        nodes={
            "select_information_need": NodeSpec(
                node=SelectInformationNeedNode(),
                output_summary=active_need_summary,
            ),
            "classify_information_need": NodeSpec(
                node=ClassifyInformationNeedNode(query_classifier),
                input_summary=active_need_summary,
                output_summary=active_need_classification_summary,
                trace_metadata=active_need_metadata,
            ),
            "plan_information_need": NodeSpec(
                node=PlanInformationNeedNode(
                    retrieval_planner,
                    available_pipeline_names=retrieval_executor.available_pipeline_names,
                    max_total_attempts=max_total_retrieval_attempts,
                ),
                input_summary=active_need_grade_summary,
                output_summary=active_need_plan_summary,
                trace_metadata=active_need_metadata,
            ),
            "execute_information_need_plan": NodeSpec(
                node=ExecuteInformationNeedPlanNode(
                    retrieval_executor,
                    max_total_attempts=max_total_retrieval_attempts,
                    max_accumulated_evidence=max_accumulated_evidence,
                ),
                input_summary=active_need_plan_summary,
                output_summary=active_need_lookup_summary,
                trace_metadata=active_need_lookup_metadata,
            ),
            "validate_information_need_constraints": NodeSpec(
                node=ValidateInformationNeedConstraintsNode(),
                input_summary=active_need_lookup_summary,
                output_summary=active_need_constraint_validation_summary,
                trace_metadata=active_need_metadata,
            ),
            "grade_information_need": NodeSpec(
                node=GradeInformationNeedNode(evidence_grader),
                input_summary=active_need_constraint_validation_summary,
                output_summary=active_need_grade_summary,
                trace_metadata=active_need_metadata,
            ),
            "decide_information_need": NodeSpec(
                node=DecideInformationNeedNode(
                    retry_policy,
                    max_total_attempts=max_total_retrieval_attempts,
                    max_reclassifications=max_reclassifications_per_information_need,
                ),
                input_summary=active_need_grade_summary,
                output_summary=active_need_decision_summary,
                trace_metadata=active_need_metadata,
            ),
            "detect_primary_document": NodeSpec(
                node=DetectPrimaryDocumentNode(primary_document_detector),
                input_summary=active_need_decision_summary,
                output_summary=primary_document_detection_summary,
                trace_metadata=primary_document_detection_metadata,
            ),
            "complete_information_need": NodeSpec(
                node=CompleteInformationNeedNode(),
                input_summary=active_need_decision_summary,
                output_summary=completed_need_summary,
                trace_metadata=completed_need_metadata,
            ),
        },
        edges={
            "select_information_need": ConditionalEdge(
                resolver=lambda state: "done" if state.active_information_need_id is None else "active",
                routes={"active": "classify_information_need", "done": END},
            ),
            "classify_information_need": "plan_information_need",
            "plan_information_need": ConditionalEdge(
                resolver=route_after_information_need_planning,
                routes={"execute": "execute_information_need_plan", "complete": "complete_information_need"},
            ),
            "execute_information_need_plan": "validate_information_need_constraints",
            "validate_information_need_constraints": "grade_information_need",
            "grade_information_need": "decide_information_need",
            "decide_information_need": ConditionalEdge(
                resolver=lambda state: active_route(state).value,
                routes={
                    InformationNeedRoute.RETRY.value: "plan_information_need",
                    InformationNeedRoute.RECLASSIFY.value: "classify_information_need",
                    InformationNeedRoute.COMPLETE_SUPPORTED.value: "detect_primary_document",
                    InformationNeedRoute.COMPLETE_EXHAUSTED.value: "detect_primary_document",
                },
            ),
            "detect_primary_document": "complete_information_need",
            "complete_information_need": "select_information_need",
        },
    )


def route_after_information_need_planning(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None:
        raise RuntimeError("Information-need planning routing requires an active work item.")
    return "execute" if execution.current_plan is not None else "complete"


def active_route(state: QueryState) -> InformationNeedRoute:
    execution = state.active_information_need_execution
    if execution is None or execution.next_route is None:
        raise RuntimeError("The active information need has no graph route decision.")
    return execution.next_route
