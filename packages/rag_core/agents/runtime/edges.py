from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from packages.rag_core.agents.runtime.protocols import RouteResolver

if TYPE_CHECKING:
    from packages.rag_core.agents.query_graph.state import QueryState

END = "__end__"


@dataclass(frozen=True, slots=True)
class ConditionalEdge:
    """Resolve a route label and map it to the next named node."""

    resolver: RouteResolver
    routes: Mapping[str, str]

    def resolve(self, state: QueryState) -> str:
        route = self.resolver(state)
        try:
            return self.routes[route]
        except KeyError as exc:
            available = ", ".join(sorted(self.routes))
            raise RuntimeError(
                f"Graph route {route!r} is not registered. Available routes: {available}.",
            ) from exc
