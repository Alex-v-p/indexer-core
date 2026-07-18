from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.rag_core.agents.information_need_graph.models import InformationNeedExecution
from packages.rag_core.agents.information_need_graph.routes import InformationNeedExecutionStatus


@dataclass(frozen=True, slots=True)
class InformationNeedResolutionReport:
    """Serializable result of the bounded information-need work queue."""

    graph_name: str
    executions: tuple[InformationNeedExecution, ...]
    total_retrieval_attempts: int
    max_total_retrieval_attempts: int
    max_attempts_per_information_need: int

    def __post_init__(self) -> None:
        if not self.graph_name.strip():
            raise ValueError("graph_name must not be empty.")
        if not self.executions:
            raise ValueError("At least one information-need execution is required.")
        if self.total_retrieval_attempts < 0:
            raise ValueError("total_retrieval_attempts must not be negative.")
        if self.max_total_retrieval_attempts <= 0:
            raise ValueError("max_total_retrieval_attempts must be positive.")
        if self.max_attempts_per_information_need <= 0:
            raise ValueError("max_attempts_per_information_need must be positive.")

    @property
    def supported_information_need_ids(self) -> tuple[str, ...]:
        return tuple(
            execution.information_need.need_id
            for execution in self.executions
            if execution.status is InformationNeedExecutionStatus.SUPPORTED
        )

    @property
    def unresolved_information_need_ids(self) -> tuple[str, ...]:
        return tuple(
            execution.information_need.need_id
            for execution in self.executions
            if execution.status is not InformationNeedExecutionStatus.SUPPORTED
        )

    @property
    def complete(self) -> bool:
        return not self.unresolved_information_need_ids

    def to_metadata(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "information_need_count": len(self.executions),
            "supported_information_need_ids": list(self.supported_information_need_ids),
            "supported_information_need_count": len(self.supported_information_need_ids),
            "unresolved_information_need_ids": list(self.unresolved_information_need_ids),
            "unresolved_information_need_count": len(self.unresolved_information_need_ids),
            "complete": self.complete,
            "total_retrieval_attempts": self.total_retrieval_attempts,
            "max_total_retrieval_attempts": self.max_total_retrieval_attempts,
            "max_attempts_per_information_need": self.max_attempts_per_information_need,
            "executions": [execution.to_metadata() for execution in self.executions],
        }
