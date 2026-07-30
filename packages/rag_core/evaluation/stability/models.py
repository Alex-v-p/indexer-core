from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.rag_core.evaluation.models import MetricValue


@dataclass(frozen=True, slots=True)
class StructuredDiagnosticSnapshot:
    stage: str
    outcome: str
    failure_code: str | None
    attempt_count: int
    repair_attempted: bool


@dataclass(frozen=True, slots=True)
class StabilityAttemptSnapshot:
    attempt_number: int
    status: str
    top_k: int
    answer: str | None
    answer_presentation: dict[str, Any] | None
    outcome: str | None
    route_signature: tuple[str, ...]
    evidence_identities: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    citations: tuple[dict[str, Any], ...]
    structured_diagnostics: tuple[StructuredDiagnosticSnapshot, ...]
    runtime_profile: dict[str, Any]
    duration_ms: int
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class StructuredStageRates:
    stage: str
    observation_count: int
    repair_rate: MetricValue
    fallback_rate: MetricValue


@dataclass(frozen=True, slots=True)
class StabilityMetrics:
    technical_success_rate: MetricValue
    outcome_consistency: MetricValue
    route_signature_consistency: MetricValue
    evidence_exact_set_agreement: MetricValue
    evidence_mean_pairwise_jaccard: MetricValue
    normalized_answer_exact_match_rate: MetricValue
    normalized_answer_mean_pairwise_token_jaccard: MetricValue
    presentation_signature_consistency: MetricValue
    structured_stage_rates: tuple[StructuredStageRates, ...] = ()


@dataclass(frozen=True, slots=True)
class StabilityCaseResult:
    case_id: str
    question: str
    top_k: int
    repetitions: int
    attempts: tuple[StabilityAttemptSnapshot, ...]
    metrics: StabilityMetrics
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StabilityEvaluationReport:
    schema_version: str
    report_type: str
    dataset_schema_version: str
    dataset_name: str
    dataset_version: str
    pipeline_name: str
    pipeline_version: str
    repetitions: int
    started_at: str
    completed_at: str
    duration_ms: int
    total_cases: int
    total_attempts: int
    succeeded_attempts: int
    failed_attempts: int
    runtime_profile: dict[str, Any]
    metrics: StabilityMetrics
    cases: tuple[StabilityCaseResult, ...]
    dataset_description: str | None = None
    dataset_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("Unsupported stability report schema version.")
        if self.report_type != "repeated_query_stability":
            raise ValueError("Unsupported stability report type.")
        if self.repetitions < 2:
            raise ValueError("Stability reports require at least two repetitions.")
