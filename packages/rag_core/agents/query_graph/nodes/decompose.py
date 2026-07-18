from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposer


class DecomposeInformationNeedsNode:
    """Graph node that extracts independently gradable answer requirements."""

    name = "decompose_information_needs"
    step_type = "query_decomposition"

    def __init__(self, decomposer: InformationNeedDecomposer) -> None:
        self._decomposer = decomposer

    async def __call__(self, state: QueryState) -> QueryState:
        decomposition = await self._decomposer.decompose(state.question)
        state.information_need_decomposition = decomposition
        state.metadata["information_need_decomposition"] = decomposition.to_metadata()
        return state
