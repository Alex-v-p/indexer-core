from __future__ import annotations

from packages.rag_core.agents.state import QueryState
from packages.rag_core.retrieval.retrievers import Retriever


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
        candidate_k = state.top_k * self._candidate_multiplier
        if self._max_candidates is not None:
            candidate_k = min(candidate_k, self._max_candidates)
        candidate_k = max(state.top_k, candidate_k)

        state.retrieved_evidence = await self._retriever.retrieve(state.question, top_k=candidate_k)
        state.metadata["retrieval"] = {
            "requested_top_k": state.top_k,
            "candidate_top_k": candidate_k,
            "retrieved_count": len(state.retrieved_evidence),
        }
        return state
