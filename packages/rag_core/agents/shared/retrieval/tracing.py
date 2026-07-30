from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState


def evidence_summary(state: QueryState) -> str:
    return f"evidence_count={len(state.retrieved_evidence)}"


def rerank_input_summary(state: QueryState) -> str:
    return f"candidate_count={len(state.retrieved_evidence)}; final_top_k={state.top_k}"


def retrieval_plan_summary(state: QueryState) -> str:
    plan = state.effective_retrieval_plan
    if plan is None:
        return "retrieval_plan=missing"
    return (
        f"strategy={plan.strategy.value}; selected_pipeline={plan.selected_pipeline_name}; "
        f"target_information_needs={len(plan.target_information_need_ids)}; reranking={plan.requires_reranking}"
    )


def planned_retrieval_summary(state: QueryState) -> str:
    execution = state.metadata.get("retrieval_plan_execution", {})
    selected = execution.get("selected_pipeline_name", "missing") if isinstance(execution, dict) else "missing"
    reranked = execution.get("reranking_applied", False) if isinstance(execution, dict) else False
    return f"selected_pipeline={selected}; evidence_count={len(state.retrieved_evidence)}; reranking_applied={reranked}"


def evidence_grading_input_summary(state: QueryState) -> str:
    return f"evidence_count={len(state.retrieved_evidence)}"


def evidence_grading_summary(state: QueryState) -> str:
    report = state.evidence_grading
    if report is None:
        return "evidence_grading=missing"
    return (
        f"status={report.status.value}; coverage={report.coverage_score:.2f}; "
        f"relevant={report.relevant_count}/{report.total_count}; "
        f"information_needs_supported={report.supported_required_information_need_count}/"
        f"{report.required_information_need_count}; answerable={report.answerable}; "
        f"partial_answer={report.partial_answer_available}; unresolved={len(report.unresolved_information)}"
    )


def evidence_context_summary(state: QueryState) -> str:
    report = state.constraint_validation
    if report is None:
        return f"evidence_count={len(state.retrieved_evidence)}; constraint_validation=missing"
    return (
        f"evidence_count={len(state.retrieved_evidence)}; "
        f"constraint_status={report.status.value}; rejected={report.rejected_count}"
    )


def evidence_context_trace_metadata(state: QueryState) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if state.constraint_validation is not None:
        metadata["constraint_validation"] = state.constraint_validation.to_metadata()
    if state.evidence_context is not None:
        metadata["evidence_context"] = state.evidence_context.to_metadata()
    return metadata
