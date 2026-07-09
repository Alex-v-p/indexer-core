from __future__ import annotations

from packages.rag_core.agents.state import QueryState
from packages.rag_core.retrieval.retrievers import Retriever


class RetrieveNode:
    """Graph node that retrieves candidate evidence for the question."""

    name = "retrieve"
    step_type = "retrieval"

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def __call__(self, state: QueryState) -> QueryState:
        state.retrieved_evidence = await self._retriever.retrieve(state.question, top_k=state.top_k)
        return state
