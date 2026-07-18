from __future__ import annotations

from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.runtime.models import NodeSpec
from packages.rag_core.agents.runtime.protocols import StateMetadata


def next_step_order(state: QueryState) -> int:
    return max((event.step_order for event in state.trace), default=0) + 1


def node_trace_metadata(
    spec: NodeSpec,
    state: QueryState,
    *,
    graph_name: str,
    graph_version: str,
    graph_depth: int,
    graph_trace_metadata: StateMetadata | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "pipeline_name": state.pipeline_name,
        "pipeline_version": state.pipeline_version,
        "graph_name": graph_name,
        "graph_version": graph_version,
        "graph_depth": graph_depth,
    }
    if graph_trace_metadata is not None:
        metadata.update(graph_trace_metadata(state))
    if spec.trace_metadata is not None:
        metadata.update(spec.trace_metadata(state))
    return metadata
