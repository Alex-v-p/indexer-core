from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from packages.rag_core.agents.query_graph.state import QueryState


class GraphNode(Protocol):
    """A node that mutates and returns QueryState."""

    name: str
    step_type: str

    async def __call__(self, state: QueryState) -> QueryState:
        """Run the node."""


StateSummary = Callable[["QueryState"], str | None]
StateMetadata = Callable[["QueryState"], dict[str, Any]]
RouteResolver = Callable[["QueryState"], str]
