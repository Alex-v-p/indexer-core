from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.query_understanding.planning import InformationNeedRetrievalPlan
from packages.rag_core.retrieval.graders import EvidenceGradingReport, InformationNeedGrade


class InformationNeedExecutionStatus(StrEnum):
    """Lifecycle state for one independently resolved information need."""

    PENDING = "pending"
    ACTIVE = "active"
    SUPPORTED = "supported"
    EXHAUSTED = "exhausted"
    FAILED = "failed"


class InformationNeedRoute(StrEnum):
    """Next route chosen by the bounded information-need controller."""

    RETRY = "retry"
    RECLASSIFY = "reclassify"
    COMPLETE_SUPPORTED = "complete_supported"
    COMPLETE_EXHAUSTED = "complete_exhausted"


@dataclass(frozen=True, slots=True)
class InformationNeedAttempt:
    """One complete plan, retrieval, and grading cycle for an information need."""

    attempt_number: int
    plan: InformationNeedRetrievalPlan
    grading: EvidenceGradingReport
    retrieved_count: int
    unique_evidence_added: int
    evidence_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.attempt_number <= 0:
            raise ValueError("attempt_number must be positive.")
        if self.attempt_number != self.plan.attempt_number:
            raise ValueError("attempt_number must match plan.attempt_number.")
        if self.retrieved_count < 0 or self.unique_evidence_added < 0:
            raise ValueError("evidence counts must not be negative.")
        if self.unique_evidence_added > self.retrieved_count:
            raise ValueError("unique_evidence_added cannot exceed retrieved_count.")
        if len(self.evidence_keys) != len(set(self.evidence_keys)):
            raise ValueError("evidence_keys must be unique.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "query": self.plan.query,
            "top_k": self.plan.top_k,
            "pipeline_name": self.plan.selected_pipeline_name,
            "strategy": self.plan.strategy.value,
            "adjustments": list(self.plan.adjustments),
            "retrieved_count": self.retrieved_count,
            "unique_evidence_added": self.unique_evidence_added,
            "evidence_keys": list(self.evidence_keys),
            "plan": self.plan.to_metadata(),
            "evidence_grading": self.grading.to_metadata(),
        }


@dataclass(slots=True)
class InformationNeedExecution:
    """Mutable execution state owned by the information-need subgraph."""

    information_need: InformationNeed
    max_attempts: int
    status: InformationNeedExecutionStatus = InformationNeedExecutionStatus.PENDING
    classification: QueryClassification | None = None
    classification_history: list[QueryClassification] = field(default_factory=list)
    current_plan: InformationNeedRetrievalPlan | None = None
    plan_history: list[InformationNeedRetrievalPlan] = field(default_factory=list)
    attempts: list[InformationNeedAttempt] = field(default_factory=list)
    evidence_keys: list[str] = field(default_factory=list)
    final_grade: InformationNeedGrade | None = None
    last_grading: EvidenceGradingReport | None = None
    next_route: InformationNeedRoute | None = None
    stop_reason: str | None = None
    stop_rationale: str | None = None
    reclassifications_used: int = 0
    parent_information_need_id: str | None = None
    depth: int = 0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        if self.depth < 0:
            raise ValueError("depth must not be negative.")

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def supported(self) -> bool:
        return self.status is InformationNeedExecutionStatus.SUPPORTED

    @property
    def unresolved(self) -> bool:
        return self.status in {
            InformationNeedExecutionStatus.EXHAUSTED,
            InformationNeedExecutionStatus.FAILED,
        }

    def add_evidence_key(self, key: str) -> None:
        if key not in self.evidence_keys:
            self.evidence_keys.append(key)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "information_need": self.information_need.to_metadata(),
            "information_need_id": self.information_need.need_id,
            "status": self.status.value,
            "attempts_used": self.attempts_used,
            "max_attempts": self.max_attempts,
            "reclassifications_used": self.reclassifications_used,
            "parent_information_need_id": self.parent_information_need_id,
            "depth": self.depth,
            "classification": self.classification.to_metadata() if self.classification is not None else None,
            "classification_history": [item.to_metadata() for item in self.classification_history],
            "current_plan": self.current_plan.to_metadata() if self.current_plan is not None else None,
            "plan_history": [item.to_metadata() for item in self.plan_history],
            "attempts": [attempt.to_metadata() for attempt in self.attempts],
            "evidence_keys": list(self.evidence_keys),
            "final_grade": self.final_grade.to_metadata() if self.final_grade is not None else None,
            "stop_reason": self.stop_reason,
            "stop_rationale": self.stop_rationale,
        }


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
