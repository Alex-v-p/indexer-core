from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState


def active_information_need_trace_metadata(state: QueryState) -> dict[str, object]:
    execution = state.active_information_need_execution
    if execution is None:
        return {}
    return {
        "information_need_id": execution.information_need.need_id,
        "parent_information_need_id": execution.parent_information_need_id,
        "information_need_depth": execution.depth,
        "information_need_attempts_used": execution.attempts_used,
    }


def active_need_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None:
        return f"active_information_need=none; pending={len(state.pending_information_need_ids)}"
    return (
        f"information_need_id={execution.information_need.need_id}; "
        f"description={execution.information_need.description!r}; attempts={execution.attempts_used}/{execution.max_attempts}"
    )


def active_need_classification_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None or execution.classification is None:
        return "information_need_classification=missing"
    return (
        f"information_need_id={execution.information_need.need_id}; "
        f"query_type={execution.classification.query_type.value}; "
        f"confidence={execution.classification.confidence:.2f}; "
        f"classification_count={len(execution.classification_history)}"
    )


def active_need_plan_summary(state: QueryState) -> str:
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


def active_need_lookup_summary(state: QueryState) -> str:
    lookup = state.metadata.get("active_information_need_lookup")
    if not isinstance(lookup, dict):
        return "information_need_lookup=missing"
    return (
        f"information_need_id={lookup.get('information_need_id')}; attempt={lookup.get('attempt_number')}; "
        f"retrieved={lookup.get('retrieved_count')}; unique_added={lookup.get('unique_evidence_added')}"
    )


def active_need_grade_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None or execution.final_grade is None:
        return "information_need_grade=pending"
    return (
        f"information_need_id={execution.information_need.need_id}; status={execution.final_grade.status.value}; "
        f"coverage={execution.final_grade.coverage_score:.2f}; attempts={execution.attempts_used}/{execution.max_attempts}"
    )


def active_need_decision_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None or execution.next_route is None:
        return "information_need_decision=missing"
    return (
        f"information_need_id={execution.information_need.need_id}; route={execution.next_route.value}; "
        f"stop_reason={execution.stop_reason or 'none'}"
    )


def active_need_metadata(state: QueryState) -> dict[str, object]:
    execution = state.active_information_need_execution
    return {"information_need_execution": execution.to_metadata()} if execution is not None else {}


def completed_need_summary(state: QueryState) -> str:
    completed = state.metadata.get("last_completed_information_need")
    if not isinstance(completed, dict):
        return f"completed_information_need=missing; pending={len(state.pending_information_need_ids)}"
    return (
        f"information_need_id={completed.get('information_need_id')}; "
        f"status={completed.get('status')}; attempts={completed.get('attempts_used')}; "
        f"pending={len(state.pending_information_need_ids)}"
    )


def completed_need_metadata(state: QueryState) -> dict[str, object]:
    completed = state.metadata.get("last_completed_information_need")
    return {"information_need_execution": completed} if isinstance(completed, dict) else {}


def active_need_constraint_validation_summary(state: QueryState) -> str:
    execution = state.active_information_need_execution
    if execution is None or execution.last_constraint_validation is None:
        return "information_need_constraint_validation=missing"
    report = execution.last_constraint_validation
    return (
        f"information_need_id={execution.information_need.need_id}; "
        f"constraint_status={report.status.value}; matched={report.matched_count}; rejected={report.rejected_count}"
    )
