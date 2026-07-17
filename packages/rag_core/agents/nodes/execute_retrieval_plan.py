from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from packages.rag_core.agents.nodes.rerank import RerankNode
from packages.rag_core.agents.nodes.retrieve import RetrieveNode
from packages.rag_core.agents.state import QueryState
from packages.rag_core.query_understanding.planning import RetrievalStrategy
from packages.rag_core.retrieval.rerankers import Reranker
from packages.rag_core.retrieval.retrievers import Retriever


@dataclass(frozen=True, slots=True)
class RetrievalPlanExecution:
    """Runtime dependencies needed to execute one selectable retrieval pipeline."""

    pipeline_name: str
    pipeline_version: str
    strategy: RetrievalStrategy
    retriever: Retriever
    reranker: Reranker | None = None
    candidate_multiplier: int = 1
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.pipeline_name.strip():
            raise ValueError("pipeline_name must not be empty.")
        if not self.pipeline_version.strip():
            raise ValueError("pipeline_version must not be empty.")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")
        expected_reranker = self.strategy is RetrievalStrategy.RERANK
        if (self.reranker is not None) is not expected_reranker:
            raise ValueError("reranker configuration must match whether strategy is rerank.")


class ExecuteRetrievalPlanNode:
    """Dispatch the planned strategy to its retriever and optional reranker."""

    name = "execute_retrieval_plan"
    step_type = "retrieval"

    def __init__(self, executions: Mapping[str, RetrievalPlanExecution]) -> None:
        if not executions:
            raise ValueError("At least one retrieval plan execution must be configured.")
        normalized: dict[str, RetrievalPlanExecution] = {}
        for pipeline_name, execution in executions.items():
            if pipeline_name != execution.pipeline_name:
                raise ValueError("Execution mapping keys must match execution.pipeline_name.")
            if pipeline_name in normalized:
                raise ValueError(f"Duplicate retrieval execution for {pipeline_name!r}.")
            normalized[pipeline_name] = execution
        self._executions = MappingProxyType(normalized)

    async def __call__(self, state: QueryState) -> QueryState:
        if state.retrieval_plan is None:
            raise RuntimeError("Retrieval plan execution requires plan_retrieval to run first.")

        selected_name = state.retrieval_plan.selected_pipeline_name
        try:
            execution = self._executions[selected_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._executions))
            raise RuntimeError(
                f"Retrieval plan selected unavailable pipeline {selected_name!r}. Available: {available}.",
            ) from exc

        if execution.strategy is not state.retrieval_plan.strategy:
            raise RuntimeError(
                f"Retrieval execution for {selected_name!r} is configured as {execution.strategy.value!r}, "
                f"but the plan selected {state.retrieval_plan.strategy.value!r}.",
            )
        if (execution.reranker is not None) is not state.retrieval_plan.requires_reranking:
            raise RuntimeError(
                f"Retrieval execution for {selected_name!r} does not match the plan's reranking requirement.",
            )

        state = await RetrieveNode(
            execution.retriever,
            candidate_multiplier=execution.candidate_multiplier,
            max_candidates=execution.max_candidates,
        )(state)
        if execution.reranker is not None:
            state = await RerankNode(execution.reranker)(state)

        state.metadata["retrieval_plan_execution"] = {
            "selected_pipeline_name": execution.pipeline_name,
            "selected_pipeline_version": execution.pipeline_version,
            "strategy": state.retrieval_plan.strategy.value,
            "reranking_applied": execution.reranker is not None,
            "retrieved_count": len(state.retrieved_evidence),
        }
        return state
