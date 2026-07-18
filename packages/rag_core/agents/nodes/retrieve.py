from __future__ import annotations

from packages.rag_core.agents.state import QueryState
from packages.rag_core.retrieval.retrievers import RetrievalBatch, Retriever


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

        retrieval_metadata: dict[str, object] = {}
        retrieve_with_metadata = getattr(self._retriever, "retrieve_with_metadata", None)
        if callable(retrieve_with_metadata):
            batch = await retrieve_with_metadata(retrieval_query, top_k=candidate_k)
            if not isinstance(batch, RetrievalBatch):
                raise TypeError("retrieve_with_metadata must return RetrievalBatch.")
            state.retrieved_evidence = batch.evidence
            retrieval_metadata = dict(batch.metadata)
        else:
            state.retrieved_evidence = await self._retriever.retrieve(retrieval_query, top_k=candidate_k)

        state.metadata["retrieval"] = {
            **retrieval_metadata,
            "query": retrieval_query,
            "requested_top_k": retrieval_top_k,
            "candidate_top_k": candidate_k,
            "retrieved_count": len(state.retrieved_evidence),
        }
        return state
