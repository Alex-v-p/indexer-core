from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.agents.runtime.protocols import GraphNode, StateMetadata, StateSummary


@dataclass(slots=True)
class TraceEvent:
    """Runtime trace emitted by a graph runner."""

    step_order: int
    name: str
    step_type: str
    status: str
    duration_ms: int | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphProgressEvent:
    """Lightweight execution update emitted around graph-node execution."""

    node_name: str
    step_type: str
    status: str
    graph_name: str
    graph_version: str
    graph_depth: int
    step_order: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Registered graph node with optional trace summarizers."""

    node: GraphNode
    input_summary: StateSummary | None = None
    output_summary: StateSummary | None = None
    trace_metadata: StateMetadata | None = None
