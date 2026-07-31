from __future__ import annotations

import time

from typing import TYPE_CHECKING

from packages.rag_core.agents.runtime.models import GraphProgressEvent, NodeSpec, TraceEvent

if TYPE_CHECKING:
    from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.runtime.protocols import StateMetadata
from packages.rag_core.agents.runtime.tracing import next_step_order, node_trace_metadata


async def run_node_spec(
    spec: NodeSpec,
    state: QueryState,
    *,
    graph_name: str,
    graph_version: str,
    graph_depth: int,
    graph_trace_metadata: StateMetadata | None = None,
) -> QueryState:
    started = time.perf_counter()
    input_summary = spec.input_summary(state) if spec.input_summary else None
    step_order = next_step_order(state)
    await _report_progress(
        state,
        GraphProgressEvent(
            node_name=spec.node.name,
            step_type=spec.node.step_type,
            status="started",
            graph_name=graph_name,
            graph_version=graph_version,
            graph_depth=graph_depth,
            step_order=step_order,
            metadata=_progress_metadata(state),
        ),
    )
    try:
        state = await spec.node(state)
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        state.error_message = str(exc)
        state.trace.append(
            TraceEvent(
                step_order=next_step_order(state),
                name=spec.node.name,
                step_type=spec.node.step_type,
                status="failed",
                duration_ms=duration_ms,
                input_summary=input_summary,
                error_message=str(exc),
                metadata=node_trace_metadata(
                    spec,
                    state,
                    graph_name=graph_name,
                    graph_version=graph_version,
                    graph_depth=graph_depth,
                    graph_trace_metadata=graph_trace_metadata,
                ),
            ),
        )
        await _report_progress(
            state,
            GraphProgressEvent(
                node_name=spec.node.name,
                step_type=spec.node.step_type,
                status="failed",
                graph_name=graph_name,
                graph_version=graph_version,
                graph_depth=graph_depth,
                step_order=step_order,
                metadata=_progress_metadata(state),
            ),
        )
        raise

    duration_ms = int((time.perf_counter() - started) * 1000)
    output_summary = spec.output_summary(state) if spec.output_summary else None
    state.trace.append(
        TraceEvent(
            step_order=next_step_order(state),
            name=spec.node.name,
            step_type=spec.node.step_type,
            status="succeeded",
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=node_trace_metadata(
                spec,
                state,
                graph_name=graph_name,
                graph_version=graph_version,
                graph_depth=graph_depth,
                graph_trace_metadata=graph_trace_metadata,
            ),
        ),
    )
    await _report_progress(
        state,
        GraphProgressEvent(
            node_name=spec.node.name,
            step_type=spec.node.step_type,
            status="succeeded",
            graph_name=graph_name,
            graph_version=graph_version,
            graph_depth=graph_depth,
            step_order=step_order,
            metadata=_progress_metadata(state),
        ),
    )
    return state


async def _report_progress(state: QueryState, event: GraphProgressEvent) -> None:
    observer = state.progress_observer
    if observer is None:
        return
    try:
        await observer(event)
    except Exception:
        # Progress reporting is telemetry. It must never change graph behavior
        # or replace the original node failure with an observer failure.
        return


def _progress_metadata(state: QueryState) -> dict[str, object]:
    executions = list(state.information_need_executions.values())
    terminal_statuses = {"supported", "exhausted", "failed"}
    completed_count = sum(
        execution.status.value in terminal_statuses
        for execution in executions
    )
    active_execution = state.active_information_need_execution
    active_index = next(
        (
            index
            for index, execution in enumerate(executions, start=1)
            if execution.information_need.need_id == state.active_information_need_id
        ),
        None,
    )
    return {
        "active_information_need_id": state.active_information_need_id,
        "active_information_need_index": active_index,
        "active_information_need_attempt": (
            active_execution.attempts_used + 1 if active_execution is not None else None
        ),
        "active_information_need_max_attempts": (
            active_execution.max_attempts if active_execution is not None else None
        ),
        "pending_information_need_count": len(state.pending_information_need_ids),
        "completed_information_need_count": completed_count,
        "information_need_count": len(executions),
        "retrieval_attempt_count": state.total_information_need_retrieval_attempts,
    }
