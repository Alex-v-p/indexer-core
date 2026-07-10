from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from packages.rag_core.agents.state import QueryState


class RetrievalPipeline(Protocol):
    """Common execution boundary for every selectable retrieval pipeline."""

    name: str
    version: str

    async def run(self, state: QueryState) -> QueryState:
        """Execute the complete retrieval and answer-generation workflow."""


PipelineFactory = Callable[[], RetrievalPipeline]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Static metadata and dependency declarations for a pipeline."""

    name: str
    version: str
    description: str
    tool_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
