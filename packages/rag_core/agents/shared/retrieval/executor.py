from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.shared.retrieval.models import RetrievalPlanExecution
from packages.rag_core.agents.shared.retrieval.nodes.rerank import RerankNode
from packages.rag_core.agents.shared.retrieval.nodes.retrieve import RetrieveNode
from packages.rag_core.query_understanding.planning import RetrievalPlan
from packages.rag_core.retrieval.document_selection import (
    DocumentCandidateSelector,
    PassthroughDocumentCandidateSelector,
)
from packages.rag_core.retrieval.models import EvidenceItem


class RetrievalPlanExecutor:
    """Dispatch a typed plan, then apply document-aware candidate selection."""

    def __init__(
        self,
        executions: Mapping[str, RetrievalPlanExecution],
        *,
        candidate_selector: DocumentCandidateSelector | None = None,
    ) -> None:
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
        self._candidate_selector = candidate_selector or PassthroughDocumentCandidateSelector()

    @property
    def available_pipeline_names(self) -> tuple[str, ...]:
        return tuple(self._executions)

    async def execute_lookup(
        self,
        *,
        plan: RetrievalPlan,
        query: str,
        top_k: int,
    ) -> tuple[list[EvidenceItem], dict[str, object]]:
        """Execute an isolated lookup without overwriting the parent QueryState."""

        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        execution = self._resolve_execution(plan)
        lookup_state = QueryState(
            question=query,
            top_k=top_k,
            retrieval_plan=plan,
            active_retrieval_plan=plan,
            active_retrieval_query=query,
            active_retrieval_top_k=top_k,
        )
        candidate_multiplier = max(
            execution.candidate_multiplier,
            self._candidate_selector.candidate_multiplier,
        )
        max_candidates = _lowest_candidate_limit(
            execution.max_candidates,
            self._candidate_selector.max_candidates,
        )
        lookup_state = await RetrieveNode(
            execution.retriever,
            candidate_multiplier=candidate_multiplier,
            max_candidates=max_candidates,
        )(lookup_state)
        if execution.reranker is not None and lookup_state.retrieved_evidence:
            lookup_state.active_retrieval_top_k = len(lookup_state.retrieved_evidence)
            lookup_state = await RerankNode(execution.reranker)(lookup_state)
            lookup_state.active_retrieval_top_k = top_k
            reranking = lookup_state.metadata.get("reranking")
            if isinstance(reranking, dict):
                reranking["final_requested_top_k"] = top_k
                reranking["document_balancing_after_rerank"] = True

        selection = self._candidate_selector.select(
            lookup_state.retrieved_evidence,
            top_k=top_k,
            preference=plan.preferred_document,
        )
        lookup_state.retrieved_evidence = list(selection.evidence)
        lookup_state.metadata["document_balancing"] = selection.to_metadata()
        metadata: dict[str, object] = {
            "selected_pipeline_name": execution.pipeline_name,
            "selected_pipeline_version": execution.pipeline_version,
            "strategy": plan.strategy.value,
            "query": query,
            "top_k": top_k,
            "reranking_applied": execution.reranker is not None,
            "retrieved_count": len(lookup_state.retrieved_evidence),
            "retrieval": lookup_state.metadata.get("retrieval", {}),
            "document_balancing": selection.to_metadata(),
            "preferred_document": (
                plan.preferred_document.to_metadata() if plan.preferred_document is not None else None
            ),
        }
        if "reranking" in lookup_state.metadata:
            metadata["reranking"] = lookup_state.metadata["reranking"]
        return lookup_state.retrieved_evidence, metadata

    async def execute(self, state: QueryState) -> QueryState:
        plan = state.effective_retrieval_plan
        if plan is None:
            raise RuntimeError("Retrieval plan execution requires a plan first.")
        evidence, metadata = await self.execute_lookup(
            plan=plan,
            query=state.effective_retrieval_query,
            top_k=state.effective_retrieval_top_k,
        )
        state.retrieved_evidence = evidence
        state.metadata["retrieval_plan_execution"] = metadata
        return state

    def _resolve_execution(self, plan: RetrievalPlan) -> RetrievalPlanExecution:
        selected_name = plan.selected_pipeline_name
        try:
            execution = self._executions[selected_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._executions))
            raise RuntimeError(
                f"Retrieval plan selected unavailable pipeline {selected_name!r}. Available: {available}.",
            ) from exc
        if execution.strategy is not plan.strategy:
            raise RuntimeError(
                f"Retrieval execution for {selected_name!r} is configured as {execution.strategy.value!r}, "
                f"but the plan selected {plan.strategy.value!r}.",
            )
        if (execution.reranker is not None) is not plan.requires_reranking:
            raise RuntimeError(
                f"Retrieval execution for {selected_name!r} does not match the plan's reranking requirement.",
            )
        return execution


def _lowest_candidate_limit(left: int | None, right: int | None) -> int | None:
    values = tuple(value for value in (left, right) if value is not None)
    return min(values) if values else None
