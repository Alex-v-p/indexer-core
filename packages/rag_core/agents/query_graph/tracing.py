from __future__ import annotations

from typing import Any

from packages.rag_core.agents.query_graph.state import QueryState


def question_summary(state: QueryState) -> str:
    return f"question={state.question!r}; top_k={state.top_k}"


def answer_summary(state: QueryState) -> str:
    answer_length = len(state.answer or "")
    candidate_count = state.metadata.get("candidate_evidence_count", len(state.retrieved_evidence))
    filtered_count = state.metadata.get("irrelevant_evidence_filtered_count", 0)
    is_partial = state.metadata.get("answer_is_partial", False)
    return (
        f"answer_length={answer_length}; citation_count={len(state.citations)}; "
        f"answer_evidence={len(state.retrieved_evidence)}/{candidate_count}; "
        f"filtered_irrelevant={filtered_count}; partial={is_partial}"
    )


def classification_summary(state: QueryState) -> str:
    classification = state.query_classification
    if classification is None:
        return "classification=missing"
    hints = ",".join(hint.value for hint in classification.metadata_filter_hints) or "none"
    return (
        f"query_type={classification.query_type.value}; confidence={classification.confidence:.2f}; "
        f"metadata_filters={classification.needs_metadata_filters}; filter_hints={hints}; "
        f"fallback={classification.fallback_used}"
    )


def classification_trace_metadata(state: QueryState) -> dict[str, Any]:
    classification = state.query_classification
    return {"classification": classification.to_metadata()} if classification is not None else {}


def information_need_decomposition_summary(state: QueryState) -> str:
    decomposition = state.information_need_decomposition
    if decomposition is None:
        return "information_need_decomposition=missing"
    required_count = sum(1 for need in decomposition.information_needs if need.required)
    return (
        f"information_needs={len(decomposition.information_needs)}; required={required_count}; "
        f"decomposer={decomposition.decomposer_name}; fallback={decomposition.fallback_used}"
    )


def information_need_decomposition_trace_metadata(state: QueryState) -> dict[str, Any]:
    decomposition = state.information_need_decomposition
    return {"information_need_decomposition": decomposition.to_metadata()} if decomposition is not None else {}


def information_need_resolution_summary(state: QueryState) -> str:
    report = state.information_need_resolution
    if report is None:
        return "information_need_resolution=missing"
    return (
        f"supported={len(report.supported_information_need_ids)}/{len(report.executions)}; "
        f"unresolved={len(report.unresolved_information_need_ids)}; "
        f"retrieval_attempts={report.total_retrieval_attempts}/{report.max_total_retrieval_attempts}"
    )


def information_need_resolution_trace_metadata(state: QueryState) -> dict[str, Any]:
    report = state.information_need_resolution
    return {"information_need_resolution": report.to_metadata()} if report is not None else {}
