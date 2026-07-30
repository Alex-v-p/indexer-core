from __future__ import annotations

from collections.abc import Mapping

from packages.rag_core.agents.information_need_graph.graph import build_information_need_graph
from packages.rag_core.agents.query_graph.graph import build_query_graph
from packages.rag_core.agents.shared.retrieval import RetrievalPlanExecution, RetrievalPlanExecutor
from packages.rag_core.agents.runtime import GraphRunner
from packages.rag_core.pipelines.base import PipelineConfig
from packages.rag_core.pipelines.baseline import BASELINE_LLM_TOOL, BASELINE_RETRIEVER_TOOL
from packages.rag_core.pipelines.contextual import (
    CONTEXTUAL_KEYWORD_RETRIEVER_TOOL,
    CONTEXTUAL_RETRIEVER_TOOL,
    CONTEXTUAL_VECTOR_RETRIEVER_TOOL,
)
from packages.rag_core.pipelines.hierarchical import HIERARCHICAL_RETRIEVER_TOOL
from packages.rag_core.pipelines.hybrid import HYBRID_KEYWORD_RETRIEVER_TOOL, HYBRID_RETRIEVER_TOOL
from packages.rag_core.pipelines.hybrid_cross_encoder_rerank import HYBRID_CROSS_ENCODER_RERANKER_TOOL
from packages.rag_core.pipelines.multi_query import MULTI_QUERY_GENERATOR_TOOL, MULTI_QUERY_RETRIEVER_TOOL
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
from packages.rag_core.retrieval.arbitration import (
    EVIDENCE_ARBITRATOR_TOOL,
    EvidenceArbitrator,
    PriorGradeEvidenceArbitrator,
)
from packages.rag_core.retrieval.graders import EVIDENCE_GRADER_TOOL, EvidenceGrader
from packages.rag_core.retrieval.document_selection import (
    DOCUMENT_CANDIDATE_SELECTOR_TOOL,
    PRIMARY_DOCUMENT_DETECTOR_TOOL,
    DocumentCandidateSelector,
    NoPrimaryDocumentDetector,
    PassthroughDocumentCandidateSelector,
    PrimaryDocumentDetector,
)
from packages.rag_core.retrieval.retry import RETRIEVAL_RETRY_POLICY_TOOL, RetrievalRetryPolicy

AGENTIC_RAG_NAME = "agentic_rag"
AGENTIC_RAG_VERSION = "0.14.0"

AGENTIC_RAG_CONFIG = PipelineConfig(
    name=AGENTIC_RAG_NAME,
    version=AGENTIC_RAG_VERSION,
    description=(
        "Classify and decompose the query, then resolve every information need through a reusable bounded subgraph "
        "that independently classifies, plans, retrieves, validates metadata constraints, grades, and retries that item before aggregating and "
        "strictly arbitrating complete or explicitly partial answer evidence against the original question."
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
        HIERARCHICAL_RETRIEVER_TOOL,
        MULTI_QUERY_GENERATOR_TOOL,
        MULTI_QUERY_RETRIEVER_TOOL,
        HYBRID_CROSS_ENCODER_RERANKER_TOOL,
        EVIDENCE_GRADER_TOOL,
        PRIMARY_DOCUMENT_DETECTOR_TOOL,
        DOCUMENT_CANDIDATE_SELECTOR_TOOL,
        EVIDENCE_ARBITRATOR_TOOL,
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
            "arbitrate_final_evidence",
            "prepare_evidence_context",
            "generate_answer",
        ),
        "information_need_subgraph_stages": (
            "select_information_need",
            "classify_information_need",
            "plan_information_need",
            "execute_information_need_plan",
            "validate_information_need_constraints",
            "grade_information_need",
            "decide_information_need",
            "detect_primary_document",
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
        "selectable_strategies": ("baseline", "hybrid", "contextual", "hierarchical", "multi_query", "rerank"),
        "retry_mode": "per_information_need_bounded_cycles",
        "retry_budget_mode": "per_information_need_and_query_global_limits",
        "constraint_enforcement_mode": "strict_subgraph_and_pre_generation_validation",
        "evidence_metadata_context_mode": "constraint_relevant_compact_source_metadata",
        "answer_evidence_mode": "question_level_arbitrated_per_information_need_evidence",
        "document_preference_mode": "soft_primary_document_inherited_by_later_information_needs",
        "candidate_selection_mode": "document_balanced_with_primary_and_secondary_quotas",
        "partial_answer_mode": "explicit_unresolved_information_need_disclosure",
        "code_organization": "graph_owned_packages_with_shared_runtime",
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
    evidence_arbitrator: EvidenceArbitrator | None = None,
    primary_document_detector: PrimaryDocumentDetector | None = None,
    document_candidate_selector: DocumentCandidateSelector | None = None,
    max_retries_per_information_need: int = 2,
    max_total_retrieval_attempts: int = 20,
    max_accumulated_evidence: int = 40,
    max_reclassifications_per_information_need: int = 1,
) -> GraphRunner:
    """Wire the top-level query graph to its reusable information-need subgraph."""

    if max_retries_per_information_need < 0:
        raise ValueError("max_retries_per_information_need must not be negative.")
    max_attempts_per_need = max_retries_per_information_need + 1
    retrieval_executor = RetrievalPlanExecutor(
        executions,
        candidate_selector=document_candidate_selector or PassthroughDocumentCandidateSelector(),
    )
    information_need_subgraph = build_information_need_graph(
        query_classifier=query_classifier,
        retrieval_planner=retrieval_planner,
        retrieval_executor=retrieval_executor,
        evidence_grader=evidence_grader,
        primary_document_detector=primary_document_detector or NoPrimaryDocumentDetector(),
        retry_policy=retry_policy,
        max_total_retrieval_attempts=max_total_retrieval_attempts,
        max_accumulated_evidence=max_accumulated_evidence,
        max_reclassifications_per_information_need=max_reclassifications_per_information_need,
    )
    return build_query_graph(
        name=AGENTIC_RAG_NAME,
        version=AGENTIC_RAG_VERSION,
        query_classifier=query_classifier,
        information_need_decomposer=information_need_decomposer,
        information_need_subgraph=information_need_subgraph,
        evidence_arbitrator=evidence_arbitrator or PriorGradeEvidenceArbitrator(),
        llm_provider=llm_provider,
        max_attempts_per_information_need=max_attempts_per_need,
        max_total_retrieval_attempts=max_total_retrieval_attempts,
    )
