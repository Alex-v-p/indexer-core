from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MetricStatus = Literal["computed", "not_applicable", "not_implemented", "failed"]
CaseStatus = Literal["succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class EvidenceExpectation:
    """Ground-truth evidence matcher for one relevant chunk.

    Every populated field is treated as a required match. ID fields are the
    most stable option for a fixed environment, while metadata and text
    fragments are useful for portable datasets that are re-indexed often.
    """

    description: str | None = None
    qdrant_chunk_index_id: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    text_contains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        has_matcher = any(
            (
                self.qdrant_chunk_index_id,
                self.document_id,
                self.document_version_id,
                self.metadata,
                self.text_contains,
            ),
        )
        if not has_matcher:
            raise ValueError("Expected evidence must define at least one matching field.")
        if any(not fragment.strip() for fragment in self.text_contains):
            raise ValueError("Expected evidence text_contains values must not be blank.")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One question and its expected answer/evidence annotations."""

    id: str
    question: str
    expected_answer: str | None = None
    expected_evidence: tuple[EvidenceExpectation, ...] = ()
    top_k: int | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Evaluation case id must not be blank.")
        if not self.question.strip():
            raise ValueError(f"Evaluation case {self.id!r} question must not be blank.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"Evaluation case {self.id!r} top_k must be positive.")


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Versioned collection of evaluation cases."""

    schema_version: str
    name: str
    version: str
    cases: tuple[EvaluationCase, ...]
    description: str | None = None
    default_top_k: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"Unsupported evaluation dataset schema version: {self.schema_version!r}.")
        if not self.name.strip():
            raise ValueError("Evaluation dataset name must not be blank.")
        if not self.version.strip():
            raise ValueError("Evaluation dataset version must not be blank.")
        if self.default_top_k <= 0:
            raise ValueError("Evaluation dataset default_top_k must be positive.")
        if not self.cases:
            raise ValueError("Evaluation dataset must contain at least one case.")
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case ids must be unique within a dataset.")


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One metric value with an explicit computation state."""

    value: float | None
    status: MetricStatus
    details: str | None = None


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    recall_at_k: MetricValue
    reciprocal_rank: MetricValue
    citation_hit_rate: MetricValue
    answer_faithfulness: MetricValue


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    recall_at_k: MetricValue
    mrr: MetricValue
    citation_hit_rate: MetricValue
    answer_faithfulness: MetricValue


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    case_id: str
    question: str
    top_k: int
    status: CaseStatus
    duration_ms: int
    expected_answer: str | None
    expected_evidence: tuple[EvidenceExpectation, ...]
    actual_answer: str | None
    evidence: tuple[dict[str, Any], ...]
    citations: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]
    metrics: CaseMetrics
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: str
    dataset_schema_version: str
    dataset_name: str
    dataset_version: str
    pipeline_name: str
    pipeline_version: str
    started_at: str
    completed_at: str
    duration_ms: int
    total_cases: int
    succeeded_cases: int
    failed_cases: int
    metrics: AggregateMetrics
    cases: tuple[EvaluationCaseResult, ...]
    dataset_description: str | None = None
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
