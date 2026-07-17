from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from packages.rag_core.agents.state import QueryState, TraceEvent


class GraphNode(Protocol):
    """A node that mutates and returns QueryState."""

    name: str
    step_type: str

    async def __call__(self, state: QueryState) -> QueryState:
        """Run the node."""


StateSummary = Callable[[QueryState], str | None]
StateMetadata = Callable[[QueryState], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Registered graph node with optional trace summarizers."""

    node: GraphNode
    input_summary: StateSummary | None = None
    output_summary: StateSummary | None = None
    trace_metadata: StateMetadata | None = None


class GraphRunner:
    """Minimal sequential graph runner shared by selectable RAG pipelines.

    The runner stays intentionally lightweight while retaining a stable
    QueryState boundary for future conditional edges, retries, and agentic
    planning. Each run records pipeline selection before executing its nodes.
    """

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
                metadata={"pipeline_name": self.name, "pipeline_version": self.version},
            ),
        )

        for spec in self._nodes:
            step_order = _next_step_order(state)
            started = time.perf_counter()
            input_summary = spec.input_summary(state) if spec.input_summary else None
            try:
                state = await spec.node(state)
            except Exception as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                state.error_message = str(exc)
                state.trace.append(
                    TraceEvent(
                        step_order=step_order,
                        name=spec.node.name,
                        step_type=spec.node.step_type,
                        status="failed",
                        duration_ms=duration_ms,
                        input_summary=input_summary,
                        error_message=str(exc),
                        metadata=_node_trace_metadata(self, spec, state),
                    ),
                )
                raise

            duration_ms = int((time.perf_counter() - started) * 1000)
            output_summary = spec.output_summary(state) if spec.output_summary else None
            state.trace.append(
                TraceEvent(
                    step_order=step_order,
                    name=spec.node.name,
                    step_type=spec.node.step_type,
                    status="succeeded",
                    duration_ms=duration_ms,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    metadata=_node_trace_metadata(self, spec, state),
                ),
            )

        return state


def _node_trace_metadata(runner: GraphRunner, spec: NodeSpec, state: QueryState) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "pipeline_name": runner.name,
        "pipeline_version": runner.version,
    }
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
    return f"answer_length={answer_length}; citation_count={len(state.citations)}"


def classification_summary(state: QueryState) -> str:
    classification = state.query_classification
    if classification is None:
        return "classification=missing"
    hints = ",".join(hint.value for hint in classification.metadata_filter_hints) or "none"
    return (
        f"query_type={classification.query_type.value}; "
        f"confidence={classification.confidence:.2f}; "
        f"metadata_filters={classification.needs_metadata_filters}; "
        f"filter_hints={hints}; "
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
        f"information_needs={len(decomposition.information_needs)}; "
        f"required={required_count}; "
        f"decomposer={decomposition.decomposer_name}; "
        f"fallback={decomposition.fallback_used}"
    )


def information_need_decomposition_trace_metadata(state: QueryState) -> dict[str, Any]:
    decomposition = state.information_need_decomposition
    return (
        {"information_need_decomposition": decomposition.to_metadata()}
        if decomposition is not None
        else {}
    )


def retrieval_planning_input_summary(state: QueryState) -> str:
    classification = state.query_classification
    decomposition = state.information_need_decomposition
    query_type = classification.query_type.value if classification is not None else "missing"
    need_count = len(decomposition.information_needs) if decomposition is not None else 0
    return f"query_type={query_type}; information_needs={need_count}"


def retrieval_plan_summary(state: QueryState) -> str:
    plan = state.retrieval_plan
    if plan is None:
        return "retrieval_plan=missing"
    hints = ",".join(hint.value for hint in plan.metadata_filter_hints) or "none"
    return (
        f"strategy={plan.strategy.value}; "
        f"selected_pipeline={plan.selected_pipeline_name}; "
        f"target_information_needs={len(plan.target_information_need_ids)}; "
        f"reranking={plan.requires_reranking}; "
        f"metadata_filter_hints={hints}"
    )


def retrieval_plan_trace_metadata(state: QueryState) -> dict[str, Any]:
    plan = state.retrieval_plan
    return {"retrieval_plan": plan.to_metadata()} if plan is not None else {}


def planned_retrieval_summary(state: QueryState) -> str:
    execution = state.metadata.get("retrieval_plan_execution", {})
    selected = execution.get("selected_pipeline_name", "missing")
    reranked = execution.get("reranking_applied", False)
    return (
        f"selected_pipeline={selected}; "
        f"evidence_count={len(state.retrieved_evidence)}; "
        f"reranking_applied={reranked}"
    )


def planned_retrieval_trace_metadata(state: QueryState) -> dict[str, Any]:
    execution = state.metadata.get("retrieval_plan_execution")
    return {"retrieval_plan_execution": execution} if isinstance(execution, dict) else {}


def evidence_grading_input_summary(state: QueryState) -> str:
    return f"evidence_count={len(state.retrieved_evidence)}"


def evidence_grading_summary(state: QueryState) -> str:
    report = state.evidence_grading
    if report is None:
        return "evidence_grading=missing"
    return (
        f"status={report.status.value}; "
        f"coverage={report.coverage_score:.2f}; "
        f"relevant={report.relevant_count}/{report.total_count}; "
        f"information_needs_supported={report.supported_information_need_count}/"
        f"{len(report.information_need_grades)}; "
        f"unresolved={len(report.unresolved_information)}; "
        f"fallback={report.fallback_used}"
    )


def evidence_grading_trace_metadata(state: QueryState) -> dict[str, Any]:
    report = state.evidence_grading
    return {"evidence_grading": report.to_metadata()} if report is not None else {}
