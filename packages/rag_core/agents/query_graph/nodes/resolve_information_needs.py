from __future__ import annotations

from packages.rag_core.agents.runtime import ConditionalGraphRunner
from packages.rag_core.agents.query_graph.state import QueryState


class ResolveInformationNeedsNode:
    """Invoke the reusable cyclic information-need subgraph."""

    name = "resolve_information_needs"
    step_type = "subgraph"

    def __init__(self, subgraph: ConditionalGraphRunner) -> None:
        self._subgraph = subgraph

    async def __call__(self, state: QueryState) -> QueryState:
        return await self._subgraph.run(state)
