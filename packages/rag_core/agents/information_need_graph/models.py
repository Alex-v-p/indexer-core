from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.agents.information_need_graph.routes import (
    InformationNeedExecutionStatus,
    InformationNeedRoute,
)
from packages.rag_core.query_understanding.classification import QueryClassification
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.query_understanding.planning import InformationNeedRetrievalPlan
from packages.rag_core.retrieval.constraint_validation import ConstraintValidationReport
from packages.rag_core.retrieval.graders import EvidenceGrade, EvidenceGradingReport, InformationNeedGrade
from packages.rag_core.retrieval.models import EvidenceItem


@dataclass(frozen=True, slots=True)
class RetrievalExecutionMetadata:
    """UI-facing retrieval diagnostics shared by every retrieval strategy."""

    requested_top_k: int | None = None
    candidate_top_k: int | None = None
    retriever_type: str | None = None
    fusion_method: str | None = None
    vector_contribution: float | None = None
    keyword_contribution: float | None = None
    query_variants: tuple[str, ...] = ()
    multi_query_query_count: int | None = None
    multi_query_candidate_top_k_per_query: int | None = None
    multi_query_result_counts: dict[str, int] = field(default_factory=dict)
    hierarchical_document_candidate_count: int | None = None
    hierarchical_selected_document_version_ids: tuple[str, ...] = ()
    hierarchical_section_candidate_count: int | None = None
    hierarchical_selected_section_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_execution(
        cls,
        execution_metadata: dict[str, object],
        evidence: list[EvidenceItem],
    ) -> RetrievalExecutionMetadata:
        raw = _dict_value(execution_metadata.get("retrieval"))
        first_metadata = evidence[0].metadata if evidence else {}
        fusion = _dict_value(first_metadata.get("fusion"))
        fusion_sources = _dict_value(fusion.get("sources"))
        vector = _dict_value(fusion_sources.get("vector"))
        keyword = _dict_value(fusion_sources.get("keyword"))
        queries = raw.get("queries")
        query_variants = (
            tuple(
                str(item["text"])
                for item in queries
                if isinstance(item, dict)
                and item.get("kind") == "variant"
                and isinstance(item.get("text"), str)
            )
            if isinstance(queries, list)
            else ()
        )
        result_counts = _int_dict(raw.get("result_counts"))
        retriever_type = _string_value(raw.get("retriever_type") or raw.get("strategy"))
        if retriever_type is None:
            retriever_type = _string_value(first_metadata.get("retrieval_source"))
        return cls(
            requested_top_k=_int_value(raw.get("requested_top_k") or execution_metadata.get("top_k")),
            candidate_top_k=_int_value(raw.get("candidate_top_k")),
            retriever_type=retriever_type,
            fusion_method=_string_value(raw.get("fusion_method") or fusion.get("method")),
            vector_contribution=_float_value(vector.get("weight")),
            keyword_contribution=_float_value(keyword.get("weight")),
            query_variants=query_variants,
            multi_query_query_count=_int_value(raw.get("query_count")),
            multi_query_candidate_top_k_per_query=_int_value(raw.get("candidate_top_k_per_query")),
            multi_query_result_counts=result_counts,
            hierarchical_document_candidate_count=_int_value(
                raw.get("document_summary_candidate_count"),
            ),
            hierarchical_selected_document_version_ids=_string_tuple(
                raw.get("selected_document_version_ids"),
            ),
            hierarchical_section_candidate_count=_int_value(
                raw.get("section_summary_candidate_count"),
            ),
            hierarchical_selected_section_ids=_string_tuple(
                raw.get("selected_hierarchy_section_ids"),
            ),
            details=deepcopy(raw),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_top_k": self.requested_top_k,
            "candidate_top_k": self.candidate_top_k,
            "retriever_type": self.retriever_type,
            "fusion_method": self.fusion_method,
            "vector_contribution": self.vector_contribution,
            "keyword_contribution": self.keyword_contribution,
            "query_variants": list(self.query_variants),
            "multi_query_query_count": self.multi_query_query_count,
            "multi_query_candidate_top_k_per_query": self.multi_query_candidate_top_k_per_query,
            "multi_query_result_counts": dict(self.multi_query_result_counts),
            "hierarchical_document_candidate_count": self.hierarchical_document_candidate_count,
            "hierarchical_selected_document_version_ids": list(
                self.hierarchical_selected_document_version_ids,
            ),
            "hierarchical_section_candidate_count": self.hierarchical_section_candidate_count,
            "hierarchical_selected_section_ids": list(self.hierarchical_selected_section_ids),
            "details": deepcopy(self.details),
        }


@dataclass(frozen=True, slots=True)
class RerankingMetadata:
    """Structured reranking outcome for one information-need attempt."""

    applied: bool = False
    provider: str | None = None
    candidate_count_before: int | None = None
    candidate_count_after: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_execution(cls, execution_metadata: dict[str, object]) -> RerankingMetadata:
        raw = _dict_value(execution_metadata.get("reranking"))
        providers = raw.get("providers")
        provider = (
            str(providers[0])
            if isinstance(providers, list) and providers
            else _string_value(raw.get("provider"))
        )
        return cls(
            applied=bool(execution_metadata.get("reranking_applied", raw)),
            provider=provider,
            candidate_count_before=_int_value(raw.get("candidate_count")),
            candidate_count_after=_int_value(raw.get("result_count")),
            details=deepcopy(raw),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "provider": self.provider,
            "candidate_count_before": self.candidate_count_before,
            "candidate_count_after": self.candidate_count_after,
            "details": deepcopy(self.details),
        }


@dataclass(frozen=True, slots=True)
class DocumentBalancingMetadata:
    """Structured document-aware selection outcome for one attempt."""

    selector_name: str | None = None
    candidate_count: int | None = None
    requested_top_k: int | None = None
    selected_count: int | None = None
    selected_chunks_per_document: dict[str, int] = field(default_factory=dict)
    primary_document_key: str | None = None
    primary_document_quota: int | None = None
    primary_document_selected_count: int | None = None
    quota_relaxed: bool = False
    preferred_document_active: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_execution(cls, execution_metadata: dict[str, object]) -> DocumentBalancingMetadata:
        raw = _dict_value(execution_metadata.get("document_balancing"))
        return cls(
            selector_name=_string_value(raw.get("selector_name")),
            candidate_count=_int_value(raw.get("candidate_count")),
            requested_top_k=_int_value(raw.get("requested_top_k")),
            selected_count=_int_value(raw.get("selected_count")),
            selected_chunks_per_document=_int_dict(raw.get("selected_document_counts")),
            primary_document_key=_string_value(raw.get("primary_document_key")),
            primary_document_quota=_int_value(raw.get("primary_document_quota")),
            primary_document_selected_count=_int_value(raw.get("primary_selected_count")),
            quota_relaxed=bool(raw.get("quota_relaxed", False)),
            preferred_document_active=execution_metadata.get("preferred_document") is not None,
            details=deepcopy(raw),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "selector_name": self.selector_name,
            "candidate_count": self.candidate_count,
            "requested_top_k": self.requested_top_k,
            "selected_count": self.selected_count,
            "selected_chunks_per_document": dict(self.selected_chunks_per_document),
            "primary_document_key": self.primary_document_key,
            "primary_document_quota": self.primary_document_quota,
            "primary_document_selected_count": self.primary_document_selected_count,
            "quota_relaxed": self.quota_relaxed,
            "preferred_document_active": self.preferred_document_active,
            "details": deepcopy(self.details),
        }


@dataclass(frozen=True, slots=True)
class InformationNeedAttemptEvidence:
    """Immutable snapshot of one balanced retrieval result before need-level pruning."""

    evidence_key: str
    retrieval_order: int
    aggregate_rank: int | None
    text: str
    score: float | None
    qdrant_chunk_index_id: uuid.UUID | None
    document_id: uuid.UUID | None
    document_version_id: uuid.UUID | None
    metadata: dict[str, Any]
    relevance_score: float | None = None
    relevant: bool | None = None
    grading_rationale: str | None = None
    supports_information_need_ids: tuple[str, ...] = ()
    retained_after_need_grading: bool = False

    def __post_init__(self) -> None:
        if not self.evidence_key:
            raise ValueError("evidence_key must not be empty.")
        if self.retrieval_order <= 0:
            raise ValueError("retrieval_order must be positive.")
        if self.aggregate_rank is not None and self.aggregate_rank <= 0:
            raise ValueError("aggregate_rank must be positive when provided.")

    @classmethod
    def capture(
        cls,
        item: EvidenceItem,
        *,
        evidence_key: str,
        retrieval_order: int,
        aggregate_rank: int | None,
    ) -> InformationNeedAttemptEvidence:
        return cls(
            evidence_key=evidence_key,
            retrieval_order=retrieval_order,
            aggregate_rank=aggregate_rank,
            text=item.text,
            score=item.score,
            qdrant_chunk_index_id=item.qdrant_chunk_index_id,
            document_id=item.document_id,
            document_version_id=item.document_version_id,
            metadata=deepcopy(item.metadata),
        )

    def with_grading(
        self,
        grade: EvidenceGrade | None,
        *,
        retained_after_need_grading: bool,
    ) -> InformationNeedAttemptEvidence:
        return InformationNeedAttemptEvidence(
            evidence_key=self.evidence_key,
            retrieval_order=self.retrieval_order,
            aggregate_rank=self.aggregate_rank,
            text=self.text,
            score=self.score,
            qdrant_chunk_index_id=self.qdrant_chunk_index_id,
            document_id=self.document_id,
            document_version_id=self.document_version_id,
            metadata=deepcopy(self.metadata),
            relevance_score=grade.relevance_score if grade is not None else None,
            relevant=grade.relevant if grade is not None else None,
            grading_rationale=grade.rationale if grade is not None else None,
            supports_information_need_ids=(
                grade.supports_information_need_ids if grade is not None else ()
            ),
            retained_after_need_grading=retained_after_need_grading,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "evidence_key": self.evidence_key,
            "retrieval_order": self.retrieval_order,
            "aggregate_rank": self.aggregate_rank,
            "text": self.text,
            "score": self.score,
            "qdrant_chunk_index_id": (
                str(self.qdrant_chunk_index_id) if self.qdrant_chunk_index_id is not None else None
            ),
            "document_id": str(self.document_id) if self.document_id is not None else None,
            "document_version_id": (
                str(self.document_version_id) if self.document_version_id is not None else None
            ),
            "metadata": deepcopy(self.metadata),
            "relevance_score": self.relevance_score,
            "relevant": self.relevant,
            "grading_rationale": self.grading_rationale,
            "supports_information_need_ids": list(self.supports_information_need_ids),
            "retained_after_need_grading": self.retained_after_need_grading,
        }


@dataclass(frozen=True, slots=True)
class InformationNeedAttempt:
    """One complete plan, retrieval, and grading cycle for an information need."""

    attempt_number: int
    plan: InformationNeedRetrievalPlan
    grading: EvidenceGradingReport
    retrieved_count: int
    unique_evidence_added: int
    evidence_keys: tuple[str, ...]
    constraint_validation: ConstraintValidationReport
    evidence: tuple[InformationNeedAttemptEvidence, ...] = ()
    retrieval_metadata: RetrievalExecutionMetadata = field(default_factory=RetrievalExecutionMetadata)
    reranking_metadata: RerankingMetadata = field(default_factory=RerankingMetadata)
    document_balancing: DocumentBalancingMetadata = field(default_factory=DocumentBalancingMetadata)

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
        if self.evidence:
            expected_orders = list(range(1, len(self.evidence) + 1))
            if [item.retrieval_order for item in self.evidence] != expected_orders:
                raise ValueError("attempt evidence must preserve sequential retrieval order.")
            if len(self.evidence) != self.retrieved_count:
                raise ValueError("retrieved_count must match the attempt evidence snapshot.")

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
            "evidence": [item.to_metadata() for item in self.evidence],
            "retrieval_metadata": self.retrieval_metadata.to_metadata(),
            "reranking_metadata": self.reranking_metadata.to_metadata(),
            "document_balancing": self.document_balancing.to_metadata(),
            "constraint_validation": self.constraint_validation.to_metadata(),
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
    last_constraint_validation: ConstraintValidationReport | None = None
    constraint_validation_history: list[ConstraintValidationReport] = field(default_factory=list)
    next_route: InformationNeedRoute | None = None
    stop_reason: str | None = None
    stop_rationale: str | None = None
    reclassifications_used: int = 0
    parent_information_need_id: str | None = None
    depth: int = 0
    pending_attempt_evidence: tuple[InformationNeedAttemptEvidence, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    pending_retrieval_metadata: RetrievalExecutionMetadata = field(
        default_factory=RetrievalExecutionMetadata,
        repr=False,
    )
    pending_reranking_metadata: RerankingMetadata = field(
        default_factory=RerankingMetadata,
        repr=False,
    )
    pending_document_balancing: DocumentBalancingMetadata = field(
        default_factory=DocumentBalancingMetadata,
        repr=False,
    )

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
            "constraint_validation_history": [
                report.to_metadata() for report in self.constraint_validation_history
            ],
            "evidence_keys": list(self.evidence_keys),
            "final_grade": self.final_grade.to_metadata() if self.final_grade is not None else None,
            "stop_reason": self.stop_reason,
            "stop_rationale": self.stop_rationale,
        }


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else ()


def _int_dict(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        parsed = _int_value(item)
        if parsed is not None:
            result[str(key)] = parsed
    return result
