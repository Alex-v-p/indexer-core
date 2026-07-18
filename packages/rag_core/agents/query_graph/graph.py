from __future__ import annotations

from packages.rag_core.agents.query_graph.nodes import (
    AggregateInformationNeedsNode,
    ClassifyQueryNode,
    DecomposeInformationNeedsNode,
    GenerateAnswerNode,
    InitializeInformationNeedWorkNode,
    ResolveInformationNeedsNode,
)
from packages.rag_core.agents.query_graph.tracing import (
    answer_summary,
    classification_summary,
    classification_trace_metadata,
    information_need_decomposition_summary,
    information_need_decomposition_trace_metadata,
    information_need_resolution_summary,
    information_need_resolution_trace_metadata,
    question_summary,
)
from packages.rag_core.agents.runtime import ConditionalGraphRunner, GraphRunner, NodeSpec
from packages.rag_core.agents.shared.retrieval.nodes import PrepareEvidenceContextNode
from packages.rag_core.agents.shared.retrieval.tracing import (
    evidence_context_summary,
    evidence_context_trace_metadata,
)
from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.classification import HeuristicQueryClassifier, QueryClassifier
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposer


def build_query_classification_node(classifier: QueryClassifier | None = None) -> NodeSpec:
    """Build the shared first node used by every query pipeline."""

    return NodeSpec(
        node=ClassifyQueryNode(classifier or HeuristicQueryClassifier()),
        input_summary=question_summary,
        output_summary=classification_summary,
        trace_metadata=classification_trace_metadata,
    )


def build_query_graph(
    *,
    name: str,
    version: str,
    query_classifier: QueryClassifier,
    information_need_decomposer: InformationNeedDecomposer,
    information_need_subgraph: ConditionalGraphRunner,
    llm_provider: LLMProvider,
    max_attempts_per_information_need: int,
    max_total_retrieval_attempts: int,
) -> GraphRunner:
    """Build the readable top-level graph around the information-need subgraph."""

    return GraphRunner(
        name=name,
        version=version,
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
                    max_attempts_per_information_need=max_attempts_per_information_need,
                    max_total_attempts=max_total_retrieval_attempts,
                ),
                input_summary=information_need_decomposition_summary,
                output_summary=lambda state: (
                    f"queued_information_needs={len(state.pending_information_need_ids)}; "
                    f"max_attempts_per_need={max_attempts_per_information_need}; "
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
                    subgraph_name=information_need_subgraph.name,
                    max_total_attempts=max_total_retrieval_attempts,
                    max_attempts_per_information_need=max_attempts_per_information_need,
                ),
                input_summary=lambda state: f"evidence_count={len(state.retrieved_evidence)}",
                output_summary=information_need_resolution_summary,
                trace_metadata=information_need_resolution_trace_metadata,
            ),
            NodeSpec(
                node=PrepareEvidenceContextNode(),
                input_summary=information_need_resolution_summary,
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
