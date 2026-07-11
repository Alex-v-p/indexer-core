from __future__ import annotations

from packages.rag_core.agents.state import QueryState
from packages.rag_core.retrieval.rerankers import Reranker


class RerankNode:
    """Graph node that reranks retrieved candidates before answer generation."""

    name = "rerank"
    step_type = "reranking"

    def __init__(self, reranker: Reranker) -> None:
        self._reranker = reranker

    async def __call__(self, state: QueryState) -> QueryState:
        candidate_count = len(state.retrieved_evidence)
        state.retrieved_evidence = await self._reranker.rerank(
            state.question,
            state.retrieved_evidence,
            top_k=state.top_k,
        )
        state.metadata["reranking"] = {
            "candidate_count": candidate_count,
            "result_count": len(state.retrieved_evidence),
            "top_k": state.top_k,
        }
        return state
