from __future__ import annotations

import time

from typing import TYPE_CHECKING

from packages.rag_core.agents.runtime.models import NodeSpec, TraceEvent

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
    return state
