from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from packages.rag_core.agents.state import QueryState, TraceEvent

END = "__end__"


class GraphNode(Protocol):
    """A node that mutates and returns QueryState."""

    name: str
    step_type: str

    async def __call__(self, state: QueryState) -> QueryState:
        """Run the node."""


StateSummary = Callable[[QueryState], str | None]
StateMetadata = Callable[[QueryState], dict[str, Any]]
RouteResolver = Callable[[QueryState], str]


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Registered graph node with optional trace summarizers."""

    node: GraphNode
    input_summary: StateSummary | None = None
    output_summary: StateSummary | None = None
    trace_metadata: StateMetadata | None = None


@dataclass(frozen=True, slots=True)
class ConditionalEdge:
    """Resolve a route label and map it to the next named node."""

    resolver: RouteResolver
    routes: Mapping[str, str]

    def resolve(self, state: QueryState) -> str:
        route = self.resolver(state)
        try:
            return self.routes[route]
        except KeyError as exc:
            available = ", ".join(sorted(self.routes))
            raise RuntimeError(
                f"Graph route {route!r} is not registered. Available routes: {available}.",
            ) from exc


class GraphRunner:
    """Sequential top-level graph runner shared by selectable RAG pipelines."""

    def __init__(self, *, name: str, version: str, nodes: Sequence[NodeSpec]) -> None:
        if not nodes:
            raise ValueError("A graph must contain at least one node.")
        self.name = name
        self.version = version
        self._nodes = list(nodes)

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
                step_order=_next_step_order(state),
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
                    "graph_depth": 0,
                },
            ),
        )

        for spec in self._nodes:
            state = await _run_node_spec(
                spec,
                state,
                graph_name=self.name,
                graph_version=self.version,
                graph_depth=0,
            )
        return state


class ConditionalGraphRunner:
    """Named-node graph runner with bounded conditional cycles.

    It is intentionally small: nodes still operate on ``QueryState``, while the
    runner owns routing, loop limits, route validation, and graph-aware traces.
    The agentic pipeline uses it as a reusable information-need subgraph.
    """

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

            state = await _run_node_spec(
                spec,
                state,
                graph_name=self.name,
                graph_version=self.version,
                graph_depth=self._graph_depth,
            )
            edge = self._edges.get(current)
            if edge is None:
                raise RuntimeError(f"Conditional graph node {current!r} has no outgoing edge.")
            current = edge.resolve(state) if isinstance(edge, ConditionalEdge) else edge
        return state


async def _run_node_spec(
    spec: NodeSpec,
    state: QueryState,
    *,
    graph_name: str,
    graph_version: str,
    graph_depth: int,
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
                step_order=_next_step_order(state),
                name=spec.node.name,
                step_type=spec.node.step_type,
                status="failed",
                duration_ms=duration_ms,
                input_summary=input_summary,
                error_message=str(exc),
                metadata=_node_trace_metadata(
                    spec,
                    state,
                    graph_name=graph_name,
                    graph_version=graph_version,
                    graph_depth=graph_depth,
                ),
            ),
        )
        raise

    duration_ms = int((time.perf_counter() - started) * 1000)
    output_summary = spec.output_summary(state) if spec.output_summary else None
    state.trace.append(
        TraceEvent(
            step_order=_next_step_order(state),
            name=spec.node.name,
            step_type=spec.node.step_type,
            status="succeeded",
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=_node_trace_metadata(
                spec,
                state,
                graph_name=graph_name,
                graph_version=graph_version,
                graph_depth=graph_depth,
            ),
        ),
    )
    return state


def _node_trace_metadata(
    spec: NodeSpec,
    state: QueryState,
    *,
    graph_name: str,
    graph_version: str,
    graph_depth: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "pipeline_name": state.pipeline_name,
        "pipeline_version": state.pipeline_version,
        "graph_name": graph_name,
        "graph_version": graph_version,
        "graph_depth": graph_depth,
    }
    execution = state.active_information_need_execution
    if execution is not None:
        metadata.update(
            {
                "information_need_id": execution.information_need.need_id,
                "parent_information_need_id": execution.parent_information_need_id,
                "information_need_depth": execution.depth,
                "information_need_attempts_used": execution.attempts_used,
            },
        )
    if spec.trace_metadata is not None:
        metadata.update(spec.trace_metadata(state))
    return metadata


def _next_step_order(state: QueryState) -> int:
    return max((event.step_order for event in state.trace), default=0) + 1


def question_summary(state: QueryState) -> str:
    return f"question={state.question!r}; top_k={state.top_k}"


def evidence_summary(state: QueryState) -> str:
    return f"evidence_count={len(state.retrieved_evidence)}"


def rerank_input_summary(state: QueryState) -> str:
    return f"candidate_count={len(state.retrieved_evidence)}; final_top_k={state.top_k}"


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
