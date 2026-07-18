from __future__ import annotations

from collections.abc import Mapping, Sequence

from typing import TYPE_CHECKING

from packages.rag_core.agents.runtime.models import NodeSpec, TraceEvent

if TYPE_CHECKING:
    from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.runtime.edges import END, ConditionalEdge
from packages.rag_core.agents.runtime.execution import run_node_spec
from packages.rag_core.agents.runtime.protocols import StateMetadata
from packages.rag_core.agents.runtime.tracing import next_step_order


class GraphRunner:
    """Sequential top-level graph runner shared by selectable RAG pipelines."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        nodes: Sequence[NodeSpec],
        graph_depth: int = 0,
        trace_metadata: StateMetadata | None = None,
    ) -> None:
        if not nodes:
            raise ValueError("A graph must contain at least one node.")
        self.name = name
        self.version = version
        self._nodes = list(nodes)
        self._graph_depth = graph_depth
        self._trace_metadata = trace_metadata

    async def run(self, state: QueryState) -> QueryState:
        state.pipeline_name = self.name
        state.pipeline_version = self.version
        state.metadata["pipeline"] = {
            "requested_name": state.requested_pipeline_name,
            "selected_name": self.name,
            "selected_version": self.version,
        }
        state.trace.append(
            TraceEvent(
                step_order=next_step_order(state),
                name="select_pipeline",
                step_type="pipeline",
                status="succeeded",
                duration_ms=0,
                input_summary=f"requested={state.requested_pipeline_name or 'configured_default'}",
                output_summary=f"selected={self.name}@{self.version}",
                metadata={
                    "pipeline_name": self.name,
                    "pipeline_version": self.version,
                    "graph_name": self.name,
                    "graph_depth": self._graph_depth,
                },
            ),
        )

        for spec in self._nodes:
            state = await run_node_spec(
                spec,
                state,
                graph_name=self.name,
                graph_version=self.version,
                graph_depth=self._graph_depth,
                graph_trace_metadata=self._trace_metadata,
            )
        return state


class ConditionalGraphRunner:
    """Named-node graph runner with bounded conditional cycles."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        nodes: Mapping[str, NodeSpec],
        entry_point: str,
        edges: Mapping[str, str | ConditionalEdge],
        max_steps: int,
        graph_depth: int = 1,
        trace_metadata: StateMetadata | None = None,
    ) -> None:
        if not name.strip() or not version.strip():
            raise ValueError("Graph name and version must not be empty.")
        if not nodes:
            raise ValueError("A conditional graph must contain at least one node.")
        if entry_point not in nodes:
            raise ValueError("entry_point must reference a registered node.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        for node_name in edges:
            if node_name not in nodes:
                raise ValueError(f"Edge source {node_name!r} is not a registered node.")
        self.name = name
        self.version = version
        self._nodes = dict(nodes)
        self._entry_point = entry_point
        self._edges = dict(edges)
        self._max_steps = max_steps
        self._graph_depth = graph_depth
        self._trace_metadata = trace_metadata

    async def run(self, state: QueryState) -> QueryState:
        current = self._entry_point
        steps = 0
        while current != END:
            steps += 1
            if steps > self._max_steps:
                raise RuntimeError(
                    f"Conditional graph {self.name!r} exceeded its bounded step limit of {self._max_steps}.",
                )
            try:
                spec = self._nodes[current]
            except KeyError as exc:
                raise RuntimeError(f"Conditional graph selected unknown node {current!r}.") from exc

            state = await run_node_spec(
                spec,
                state,
                graph_name=self.name,
                graph_version=self.version,
                graph_depth=self._graph_depth,
                graph_trace_metadata=self._trace_metadata,
            )
            edge = self._edges.get(current)
            if edge is None:
                raise RuntimeError(f"Conditional graph node {current!r} has no outgoing edge.")
            current = edge.resolve(state) if isinstance(edge, ConditionalEdge) else edge
        return state
