from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from packages.rag_core.documents import DocumentNameConstraint, DocumentPreference, DocumentVersionConstraint
from packages.rag_core.query_understanding.temporal import DocumentDateConstraint
from packages.rag_core.query_understanding.classification import MetadataFilterHint, QueryType

if TYPE_CHECKING:
    from packages.rag_core.query_understanding.classification import QueryClassification
    from packages.rag_core.query_understanding.decomposition import InformationNeed
    from packages.rag_core.retrieval.graders.models import InformationNeedGrade


class RetrievalStrategy(StrEnum):
    """High-level retrieval behavior selected for a classified query."""

    BASELINE = "baseline"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    HIERARCHICAL = "hierarchical"
    MULTI_QUERY = "multi_query"
    RERANK = "rerank"


class ClaimSupportStatus(StrEnum):
    """Evidence status supplied to claim-level retry planning."""

    MISSING = "missing"
    PARTIAL = "partial"
    SUPPORTED = "supported"


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """Typed retrieval decision produced after query understanding."""

    strategy: RetrievalStrategy
    selected_pipeline_name: str
    rationale: str
    planner_name: str
    based_on_query_type: QueryType
    metadata_filter_hints: tuple[MetadataFilterHint, ...] = ()
    requires_reranking: bool = False
    target_information_need_ids: tuple[str, ...] = ()
    document_constraint: DocumentNameConstraint = DocumentNameConstraint()
    version_constraint: DocumentVersionConstraint = DocumentVersionConstraint()
    date_constraints: tuple[DocumentDateConstraint, ...] = ()
    preferred_document: DocumentPreference | None = None

    def __post_init__(self) -> None:
        if not self.selected_pipeline_name.strip():
            raise ValueError("selected_pipeline_name must not be empty.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.planner_name.strip():
            raise ValueError("planner_name must not be empty.")
        expected_reranking = self.strategy is RetrievalStrategy.RERANK
        if self.requires_reranking is not expected_reranking:
            raise ValueError("requires_reranking must match whether strategy is rerank.")
        normalized_ids = [need_id.strip() for need_id in self.target_information_need_ids]
        if any(not need_id for need_id in normalized_ids):
            raise ValueError("target_information_need_ids must not contain empty ids.")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("target_information_need_ids must be unique.")

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for persistence and tracing."""

        return {
            "strategy": self.strategy.value,
            "selected_pipeline_name": self.selected_pipeline_name,
            "rationale": self.rationale,
            "planner_name": self.planner_name,
            "based_on_query_type": self.based_on_query_type.value,
            "metadata_filter_hints": [hint.value for hint in self.metadata_filter_hints],
            "requires_reranking": self.requires_reranking,
            "target_information_need_ids": list(self.target_information_need_ids),
            "target_information_need_count": len(self.target_information_need_ids),
            "document_constraint": self.document_constraint.to_metadata(),
            "version_constraint": self.version_constraint.to_metadata(),
            "date_constraints": [constraint.to_metadata() for constraint in self.date_constraints],
            "preferred_document": (
                self.preferred_document.to_metadata() if self.preferred_document is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ClaimPlanningInput:
    """One unresolved answer claim and the grader feedback available for re-planning."""

    information_need_id: str
    description: str
    retrieval_query: str
    support_status: ClaimSupportStatus
    coverage_score: float
    grading_rationale: str
    supporting_evidence_ranks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.information_need_id.strip():
            raise ValueError("information_need_id must not be empty.")
        if not self.description.strip():
            raise ValueError("description must not be empty.")
        if not self.retrieval_query.strip():
            raise ValueError("retrieval_query must not be empty.")
        if self.support_status is ClaimSupportStatus.SUPPORTED:
            raise ValueError("Claim retry planning only accepts unresolved claims.")
        if not 0.0 <= self.coverage_score <= 1.0:
            raise ValueError("coverage_score must be between 0 and 1.")
        if not self.grading_rationale.strip():
            raise ValueError("grading_rationale must not be empty.")
        if any(rank <= 0 for rank in self.supporting_evidence_ranks):
            raise ValueError("supporting_evidence_ranks must be positive.")
        if len(self.supporting_evidence_ranks) != len(set(self.supporting_evidence_ranks)):
            raise ValueError("supporting_evidence_ranks must be unique.")


@dataclass(frozen=True, slots=True)
class ClaimRetrievalTask:
    """One independently executable lookup planned for an unresolved claim."""

    information_need_id: str
    description: str
    retrieval_query: str
    prior_status: ClaimSupportStatus
    prior_coverage_score: float
    prior_supporting_evidence_ranks: tuple[int, ...]
    grading_feedback: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.information_need_id.strip():
            raise ValueError("information_need_id must not be empty.")
        if not self.description.strip():
            raise ValueError("description must not be empty.")
        if not self.retrieval_query.strip():
            raise ValueError("retrieval_query must not be empty.")
        if self.prior_status is ClaimSupportStatus.SUPPORTED:
            raise ValueError("Claim retrieval tasks must target unresolved claims.")
        if not 0.0 <= self.prior_coverage_score <= 1.0:
            raise ValueError("prior_coverage_score must be between 0 and 1.")
        if any(rank <= 0 for rank in self.prior_supporting_evidence_ranks):
            raise ValueError("prior_supporting_evidence_ranks must be positive.")
        if len(self.prior_supporting_evidence_ranks) != len(set(self.prior_supporting_evidence_ranks)):
            raise ValueError("prior_supporting_evidence_ranks must be unique.")
        if not self.grading_feedback.strip():
            raise ValueError("grading_feedback must not be empty.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "information_need_id": self.information_need_id,
            "description": self.description,
            "retrieval_query": self.retrieval_query,
            "prior_status": self.prior_status.value,
            "prior_coverage_score": self.prior_coverage_score,
            "prior_supporting_evidence_ranks": list(self.prior_supporting_evidence_ranks),
            "grading_feedback": self.grading_feedback,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ClaimRetrievalPlan:
    """Focused re-planning result for the unresolved claims in one retry round."""

    tasks: tuple[ClaimRetrievalTask, ...]
    rationale: str
    planner_name: str
    deferred_information_need_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("At least one claim retrieval task is required.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.planner_name.strip():
            raise ValueError("planner_name must not be empty.")
        task_ids = [task.information_need_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Claim retrieval tasks must have unique information_need_ids.")
        deferred_ids = [need_id.strip() for need_id in self.deferred_information_need_ids]
        if any(not need_id for need_id in deferred_ids):
            raise ValueError("deferred_information_need_ids must not contain empty ids.")
        if len(deferred_ids) != len(set(deferred_ids)):
            raise ValueError("deferred_information_need_ids must be unique.")
        if set(task_ids).intersection(deferred_ids):
            raise ValueError("A claim cannot be both targeted and deferred.")

    @property
    def target_information_need_ids(self) -> tuple[str, ...]:
        return tuple(task.information_need_id for task in self.tasks)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "planner_name": self.planner_name,
            "rationale": self.rationale,
            "target_information_need_ids": list(self.target_information_need_ids),
            "target_information_need_count": len(self.tasks),
            "deferred_information_need_ids": list(self.deferred_information_need_ids),
            "deferred_information_need_count": len(self.deferred_information_need_ids),
            "tasks": [task.to_metadata() for task in self.tasks],
        }

@dataclass(frozen=True, slots=True)
class InformationNeedPlanningContext:
    """Inputs for planning one independently executable information need."""

    original_question: str
    information_need: "InformationNeed"
    classification: "QueryClassification"
    previous_grade: "InformationNeedGrade | None"
    previous_plans: tuple["InformationNeedRetrievalPlan", ...]
    previous_queries: tuple[str, ...]
    available_pipeline_names: tuple[str, ...]
    attempts_used: int
    max_attempts: int
    current_top_k: int
    preferred_document: DocumentPreference | None = None

    def __post_init__(self) -> None:
        if not self.original_question.strip():
            raise ValueError("original_question must not be empty.")
        if self.attempts_used < 0:
            raise ValueError("attempts_used must not be negative.")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        if self.attempts_used >= self.max_attempts:
            raise ValueError("Planning context cannot exceed the information-need attempt limit.")
        if self.current_top_k <= 0:
            raise ValueError("current_top_k must be positive.")
        if len(self.previous_queries) != len(set(self.previous_queries)):
            raise ValueError("previous_queries must be unique.")


@dataclass(frozen=True, slots=True)
class InformationNeedRetrievalPlan:
    """Complete executable plan for one information need and one attempt."""

    information_need_id: str
    strategy: RetrievalStrategy
    selected_pipeline_name: str
    query: str
    top_k: int
    rationale: str
    planner_name: str
    based_on_query_type: QueryType
    attempt_number: int
    metadata_filter_hints: tuple[MetadataFilterHint, ...] = ()
    requires_reranking: bool = False
    adjustments: tuple[str, ...] = ()
    document_constraint: DocumentNameConstraint = DocumentNameConstraint()
    version_constraint: DocumentVersionConstraint = DocumentVersionConstraint()
    date_constraints: tuple[DocumentDateConstraint, ...] = ()
    preferred_document: DocumentPreference | None = None

    def __post_init__(self) -> None:
        if not self.information_need_id.strip():
            raise ValueError("information_need_id must not be empty.")
        if not self.selected_pipeline_name.strip():
            raise ValueError("selected_pipeline_name must not be empty.")
        if not self.query.strip():
            raise ValueError("query must not be empty.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.planner_name.strip():
            raise ValueError("planner_name must not be empty.")
        if self.attempt_number <= 0:
            raise ValueError("attempt_number must be positive.")
        expected_reranking = self.strategy is RetrievalStrategy.RERANK
        if self.requires_reranking is not expected_reranking:
            raise ValueError("requires_reranking must match whether strategy is rerank.")
        if len(self.adjustments) != len(set(self.adjustments)):
            raise ValueError("adjustments must be unique.")

    def as_retrieval_plan(self) -> RetrievalPlan:
        """Adapt the per-need plan to the existing retrieval executor boundary."""

        return RetrievalPlan(
            strategy=self.strategy,
            selected_pipeline_name=self.selected_pipeline_name,
            rationale=self.rationale,
            planner_name=self.planner_name,
            based_on_query_type=self.based_on_query_type,
            metadata_filter_hints=self.metadata_filter_hints,
            requires_reranking=self.requires_reranking,
            target_information_need_ids=(self.information_need_id,),
            document_constraint=self.document_constraint,
            version_constraint=self.version_constraint,
            date_constraints=self.date_constraints,
            preferred_document=self.preferred_document,
        )

    @property
    def execution_signature(self) -> tuple[str, str, int]:
        return (self.selected_pipeline_name, self.query, self.top_k)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "information_need_id": self.information_need_id,
            "strategy": self.strategy.value,
            "selected_pipeline_name": self.selected_pipeline_name,
            "query": self.query,
            "top_k": self.top_k,
            "rationale": self.rationale,
            "planner_name": self.planner_name,
            "based_on_query_type": self.based_on_query_type.value,
            "attempt_number": self.attempt_number,
            "metadata_filter_hints": [hint.value for hint in self.metadata_filter_hints],
            "requires_reranking": self.requires_reranking,
            "adjustments": list(self.adjustments),
            "document_constraint": self.document_constraint.to_metadata(),
            "version_constraint": self.version_constraint.to_metadata(),
            "date_constraints": [constraint.to_metadata() for constraint in self.date_constraints],
            "preferred_document": (
                self.preferred_document.to_metadata() if self.preferred_document is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class InformationNeedPlanningStop:
    """Planner result used when another retrieval attempt would be ineffective."""

    information_need_id: str
    reason: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.information_need_id.strip():
            raise ValueError("information_need_id must not be empty.")
        if not self.reason.strip():
            raise ValueError("reason must not be empty.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "information_need_id": self.information_need_id,
            "reason": self.reason,
            "rationale": self.rationale,
        }
