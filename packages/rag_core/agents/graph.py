from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from packages.rag_core.agents.state import QueryState, TraceEvent


class GraphNode(Protocol):
    """A node that mutates and returns QueryState."""

    name: str
    step_type: str

    async def __call__(self, state: QueryState) -> QueryState:
        """Run the node."""


StateSummary = Callable[[QueryState], str | None]


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Registered graph node with optional trace summarizers."""

    node: GraphNode
    input_summary: StateSummary | None = None
    output_summary: StateSummary | None = None


class GraphRunner:
    """Minimal sequential graph runner for Phase 1.

    This intentionally stays lightweight while the first RAG path is only
    retrieve → generate_answer. Keeping it in agents/graph.py leaves a clear
    upgrade path to LangGraph-style branching, tool nodes, and conditional
    edges later without changing the API boundary.
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

        for index, spec in enumerate(self._nodes, start=1):
            started = time.perf_counter()
            input_summary = spec.input_summary(state) if spec.input_summary else None
            try:
                state = await spec.node(state)
            except Exception as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                state.error_message = str(exc)
                state.trace.append(
                    TraceEvent(
                        step_order=index,
                        name=spec.node.name,
                        step_type=spec.node.step_type,
                        status="failed",
                        duration_ms=duration_ms,
                        input_summary=input_summary,
                        error_message=str(exc),
                    ),
                )
                raise

            duration_ms = int((time.perf_counter() - started) * 1000)
            output_summary = spec.output_summary(state) if spec.output_summary else None
            state.trace.append(
                TraceEvent(
                    step_order=index,
                    name=spec.node.name,
                    step_type=spec.node.step_type,
                    status="succeeded",
                    duration_ms=duration_ms,
                    input_summary=input_summary,
                    output_summary=output_summary,
                ),
            )

        return state


def question_summary(state: QueryState) -> str:
    return f"question={state.question!r}; top_k={state.top_k}"


def evidence_summary(state: QueryState) -> str:
    return f"evidence_count={len(state.retrieved_evidence)}"


def answer_summary(state: QueryState) -> str:
    answer_length = len(state.answer or "")
    return f"answer_length={answer_length}; citation_count={len(state.citations)}"
