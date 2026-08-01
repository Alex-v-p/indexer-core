from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.retrieval.models import RetrievalConstraints
from packages.rag_core.retrieval.retrievers import RetrievalBatch, Retriever
from packages.rag_core.retrieval.retrievers.base import retrieve_batch_compatibly


class RetrieveNode:
    """Graph node that retrieves candidate evidence for the question."""

    name = "retrieve"
    step_type = "retrieval"

    def __init__(
        self,
        retriever: Retriever,
        *,
        candidate_multiplier: int = 1,
        max_candidates: int | None = None,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")

        self._retriever = retriever
        self._candidate_multiplier = candidate_multiplier
        self._max_candidates = max_candidates

    async def __call__(self, state: QueryState) -> QueryState:
        retrieval_top_k = state.effective_retrieval_top_k
        retrieval_query = state.effective_retrieval_query
        candidate_k = retrieval_top_k * self._candidate_multiplier
        if self._max_candidates is not None:
            candidate_k = min(candidate_k, self._max_candidates)
        candidate_k = max(retrieval_top_k, candidate_k)

        plan = state.effective_retrieval_plan
        document_constraint = (
            plan.document_constraint
            if plan is not None
            else (
                state.query_classification.document_constraint
                if state.query_classification is not None
                else None
            )
        )
        version_constraint = (
            plan.version_constraint
            if plan is not None
            else (
                state.query_classification.version_constraint
                if state.query_classification is not None
                else None
            )
        )
        date_constraints = (
            plan.date_constraints
            if plan is not None
            else (
                state.query_classification.date_constraints
                if state.query_classification is not None
                else ()
            )
        )
        document_scope = plan.document_scope if plan is not None else state.document_scope
        constraints = (
            RetrievalConstraints(
                document=document_constraint or RetrievalConstraints().document,
                version=version_constraint or RetrievalConstraints().version,
                dates=date_constraints,
                document_scope=document_scope,
            )
            if (
                document_constraint is not None
                or version_constraint is not None
                or date_constraints
                or document_scope.strict
            )
            else None
        )
        batch = await retrieve_batch_compatibly(
            self._retriever,
            retrieval_query,
            top_k=candidate_k,
            constraints=constraints,
        )
        state.retrieved_evidence = batch.evidence
        retrieval_metadata = dict(batch.metadata)

        state.metadata["retrieval"] = {
            **retrieval_metadata,
            "query": retrieval_query,
            "requested_top_k": retrieval_top_k,
            "candidate_top_k": candidate_k,
            "retrieved_count": len(state.retrieved_evidence),
            "constraints": constraints.to_metadata() if constraints is not None else {"active": False},
        }
        return state
