from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Sequence

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.evaluation.models import MetricValue
from packages.rag_core.retrieval.models import EvidenceItem

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_KEY_CHARACTER = re.compile(r"[^a-z0-9]+")
_SAFE_DIAGNOSTIC_OUTCOMES = {"primary_valid", "repair_valid", "fallback"}
_SENSITIVE_PROFILE_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "auth_token",
    "authorization",
    "bearer",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "id_token",
    "password",
    "passwd",
    "refresh_token",
    "secret",
    "token",
}
_SENSITIVE_PROFILE_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_auth_token",
    "_authorization",
    "_bearer_token",
    "_client_secret",
    "_credential",
    "_credentials",
    "_password",
    "_refresh_token",
    "_secret",
    "_token",
)


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


def normalize_answer(value: str | None) -> str:
    """NFKC-normalize, case-fold, and collapse Unicode whitespace."""

    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def answer_tokens(value: str | None) -> frozenset[str]:
    """Return deterministic Unicode word tokens from the normalized answer."""

    return frozenset(_TOKEN_PATTERN.findall(normalize_answer(value)))


def stable_evidence_identity(evidence: EvidenceItem) -> str:
    """Return the first available stable identity in the documented priority order."""

    if evidence.qdrant_chunk_index_id is not None:
        return f"chunk:{evidence.qdrant_chunk_index_id}"
    point_id = _first_value(evidence.metadata, ("qdrant_point_id", "point_id"))
    if point_id is not None:
        return f"point:{point_id}"
    ordinal = _first_value(
        evidence.metadata,
        ("ordinal", "chunk_ordinal", "chunk_index", "document_chunk_ordinal"),
    )
    if evidence.document_version_id is not None and ordinal is not None:
        return f"document_version_ordinal:{evidence.document_version_id}:{ordinal}"
    digest = hashlib.sha256(evidence.text.encode("utf-8")).hexdigest()
    return f"text_sha256:{digest}"


def canonical_route_signature(state: QueryState) -> tuple[str, ...]:
    """Capture only stable strategy, pipeline, top-k, stop, and status codes."""

    signature = [
        f"pipeline:{state.pipeline_name or 'unknown'}",
        f"top_k:{state.top_k}",
    ]
    metadata = state.metadata
    plan = (
        state.retrieval_plan.to_metadata()
        if state.retrieval_plan is not None
        else metadata.get("retrieval_plan")
    )
    if isinstance(plan, dict):
        signature.extend(_plan_signature("query", plan))
    retry = (
        state.retrieval_retry.to_metadata()
        if state.retrieval_retry is not None
        else metadata.get("retrieval_retry")
    )
    if isinstance(retry, dict):
        attempts = retry.get("attempts")
        if isinstance(attempts, list):
            for index, attempt in enumerate(attempts, start=1):
                if not isinstance(attempt, dict):
                    continue
                signature.extend(
                    _coded_values(
                        f"retry:attempt:{index}",
                        attempt,
                        ("strategy", "pipeline_name", "top_k"),
                    ),
                )
        signature.extend(
            _coded_values(
                "retry",
                retry,
                ("final_strategy", "final_pipeline_name", "final_top_k", "stop_reason"),
            ),
        )
    resolution = (
        state.information_need_resolution.to_metadata()
        if state.information_need_resolution is not None
        else metadata.get("information_need_resolution")
    )
    if isinstance(resolution, dict):
        executions = resolution.get("executions")
        if isinstance(executions, list):
            ordered = sorted(
                (item for item in executions if isinstance(item, dict)),
                key=lambda item: str(item.get("information_need_id") or ""),
            )
            for execution in ordered:
                need_id = str(execution.get("information_need_id") or "unknown")
                signature.extend(
                    _coded_values(
                        f"need:{need_id}",
                        execution,
                        ("status", "stop_reason"),
                    ),
                )
                plans = execution.get("plan_history")
                if isinstance(plans, list):
                    for index, item in enumerate(plans, start=1):
                        if isinstance(item, dict):
                            signature.extend(_plan_signature(f"need:{need_id}:plan:{index}", item))
    return tuple(signature)


def outcome_for_state(state: QueryState) -> str:
    presentation = state.answer_presentation
    if presentation is not None:
        return presentation.outcome.value
    blocked = bool(state.metadata.get("answer_blocked_by_evidence_grading"))
    validation = (
        state.constraint_validation.to_metadata()
        if state.constraint_validation is not None
        else state.metadata.get("constraint_validation")
    )
    if isinstance(validation, dict):
        blocked = blocked or validation.get("status") == "no_match" or validation.get("blocked") is True
    if blocked:
        return "legacy_refusal"
    return "legacy_answer" if state.answer else "legacy_no_answer"


def presentation_signature(state: QueryState) -> tuple[str, ...]:
    presentation = state.answer_presentation
    if presentation is None:
        return ("absent", outcome_for_state(state), f"body:{bool(state.answer)}")
    return (
        f"schema:{presentation.schema_version}",
        f"outcome:{presentation.outcome.value}",
        f"body:{bool(presentation.body)}",
        f"supported_section:{bool(presentation.supported_information)}",
        f"supported_count:{len(presentation.supported_information)}",
        f"unresolved_section:{bool(presentation.unresolved_information)}",
        f"unresolved_count:{len(presentation.unresolved_information)}",
        f"citation_count:{presentation.citation_count}",
    )


def safe_structured_diagnostics(metadata: dict[str, Any]) -> tuple[StructuredDiagnosticSnapshot, ...]:
    snapshots: list[StructuredDiagnosticSnapshot] = []
    _collect_diagnostics(metadata, path=(), snapshots=snapshots)
    return tuple(snapshots)


def safe_runtime_profile(state: QueryState) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "pipeline_name": state.pipeline_name,
        "pipeline_version": state.pipeline_version,
    }
    for key in ("generation_profile", "model_profile", "tool_profile", "runtime_profile"):
        value = state.metadata.get(key)
        if isinstance(value, dict):
            profile[key] = _safe_profile_value(value)
    return profile


def calculate_stability_metrics(
    attempt_groups: Sequence[Sequence[StabilityAttemptSnapshot]],
) -> StabilityMetrics:
    all_attempts = [attempt for group in attempt_groups for attempt in group]
    successful_groups = [
        [attempt for attempt in group if attempt.status == "succeeded"]
        for group in attempt_groups
    ]
    successful_attempts = [attempt for group in successful_groups for attempt in group]
    technical = _rate(
        sum(attempt.status == "succeeded" for attempt in all_attempts),
        len(all_attempts),
        empty_details="No attempts were supplied.",
    )
    return StabilityMetrics(
        technical_success_rate=technical,
        outcome_consistency=_pairwise_exact_metric(
            [[attempt.outcome for attempt in group] for group in successful_groups],
            label="successful attempt outcomes",
        ),
        route_signature_consistency=_pairwise_exact_metric(
            [[attempt.route_signature for attempt in group] for group in successful_groups],
            label="successful route signatures",
        ),
        evidence_exact_set_agreement=_pairwise_exact_metric(
            [[attempt.evidence_identities for attempt in group] for group in successful_groups],
            label="successful evidence identity sets",
        ),
        evidence_mean_pairwise_jaccard=_pairwise_similarity_metric(
            [[frozenset(attempt.evidence_identities) for attempt in group] for group in successful_groups],
            _jaccard,
            label="successful evidence identity sets",
        ),
        normalized_answer_exact_match_rate=_pairwise_exact_metric(
            [[normalize_answer(attempt.answer) for attempt in group] for group in successful_groups],
            label="successful normalized answers",
        ),
        normalized_answer_mean_pairwise_token_jaccard=_pairwise_similarity_metric(
            [[answer_tokens(attempt.answer) for attempt in group] for group in successful_groups],
            _jaccard,
            label="successful normalized answer token sets",
        ),
        presentation_signature_consistency=_pairwise_exact_metric(
            [
                [_presentation_signature_from_snapshot(attempt) for attempt in group]
                for group in successful_groups
            ],
            label="successful presentation signatures",
        ),
        structured_stage_rates=_structured_stage_rates(successful_attempts),
    )


def _presentation_signature_from_snapshot(attempt: StabilityAttemptSnapshot) -> tuple[str, ...]:
    presentation = attempt.answer_presentation
    if presentation is None:
        return ("absent", str(attempt.outcome), f"body:{bool(attempt.answer)}")
    supported = presentation.get("supported_information")
    unresolved = presentation.get("unresolved_information")
    return (
        f"schema:{presentation.get('schema_version')}",
        f"outcome:{presentation.get('outcome')}",
        f"body:{bool(presentation.get('body'))}",
        f"supported_section:{bool(supported)}",
        f"supported_count:{len(supported) if isinstance(supported, list) else 0}",
        f"unresolved_section:{bool(unresolved)}",
        f"unresolved_count:{len(unresolved) if isinstance(unresolved, list) else 0}",
        f"citation_count:{presentation.get('citation_count', 0)}",
    )


def _pairwise_exact_metric(groups: Sequence[Sequence[object]], *, label: str) -> MetricValue:
    return _pairwise_similarity_metric(
        groups,
        lambda left, right: 1.0 if left == right else 0.0,
        label=label,
    )


def _pairwise_similarity_metric(
    groups: Sequence[Sequence[object]],
    similarity,
    *,
    label: str,
) -> MetricValue:
    values: list[float] = []
    singletons = 0
    for group in groups:
        if not group:
            continue
        if len(group) == 1:
            values.append(1.0)
            singletons += 1
            continue
        values.extend(similarity(left, right) for left, right in combinations(group, 2))
    if not values:
        return MetricValue(
            value=None,
            status="not_applicable",
            details=f"No technically successful {label} were available.",
        )
    return MetricValue(
        value=sum(values) / len(values),
        status="computed",
        details=(
            f"Mean across {len(values)} within-case comparison(s); "
            f"{singletons} single-success case(s) contributed a defined value of 1.0."
        ),
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _rate(numerator: int, denominator: int, *, empty_details: str) -> MetricValue:
    if denominator == 0:
        return MetricValue(value=None, status="not_applicable", details=empty_details)
    return MetricValue(
        value=numerator / denominator,
        status="computed",
        details=f"{numerator}/{denominator}.",
    )


def _structured_stage_rates(
    attempts: Sequence[StabilityAttemptSnapshot],
) -> tuple[StructuredStageRates, ...]:
    by_stage: dict[str, list[StructuredDiagnosticSnapshot]] = {}
    for attempt in attempts:
        for diagnostic in attempt.structured_diagnostics:
            by_stage.setdefault(diagnostic.stage, []).append(diagnostic)
    return tuple(
        StructuredStageRates(
            stage=stage,
            observation_count=len(items),
            repair_rate=_rate(
                sum(item.repair_attempted for item in items),
                len(items),
                empty_details="No safe structured diagnostics were available.",
            ),
            fallback_rate=_rate(
                sum(item.outcome == "fallback" for item in items),
                len(items),
                empty_details="No safe structured diagnostics were available.",
            ),
        )
        for stage, items in sorted(by_stage.items())
    )


def _collect_diagnostics(
    value: object,
    *,
    path: tuple[str, ...],
    snapshots: list[StructuredDiagnosticSnapshot],
) -> None:
    if isinstance(value, dict):
        if _is_safe_diagnostic(value):
            stage_parts = path[:-1] if path and path[-1] == "structured_output" else path
            snapshots.append(
                StructuredDiagnosticSnapshot(
                    stage=".".join(stage_parts) or "unknown",
                    outcome=str(value["outcome"]),
                    failure_code=(
                        str(value["failure_code"]) if value.get("failure_code") is not None else None
                    ),
                    attempt_count=int(value["attempt_count"]),
                    repair_attempted=bool(value["repair_attempted"]),
                ),
            )
            return
        for key, item in value.items():
            _collect_diagnostics(item, path=(*path, str(key)), snapshots=snapshots)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_diagnostics(item, path=(*path, "[]"), snapshots=snapshots)


def _is_safe_diagnostic(value: dict[object, object]) -> bool:
    return (
        value.get("schema_version") == "1.0"
        and value.get("outcome") in _SAFE_DIAGNOSTIC_OUTCOMES
        and isinstance(value.get("attempt_count"), int)
        and not isinstance(value.get("attempt_count"), bool)
        and isinstance(value.get("repair_attempted"), bool)
    )


def _plan_signature(prefix: str, value: dict[str, Any]) -> list[str]:
    return _coded_values(prefix, value, ("strategy", "selected_pipeline_name", "top_k"))


def _coded_values(prefix: str, value: dict[str, Any], keys: Iterable[str]) -> list[str]:
    return [
        f"{prefix}:{key}:{value[key]}"
        for key in keys
        if isinstance(value.get(key), (str, int)) and not isinstance(value.get(key), bool)
    ]


def _first_value(metadata: dict[str, Any], keys: Iterable[str]) -> object | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value):
            return value
    return None


def _safe_profile_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _safe_profile_value(item)
            for key, item in value.items()
            if not _is_sensitive_profile_key(key)
            and isinstance(item, (dict, list, tuple, str, int, float, bool, type(None)))
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_profile_value(item)
            for item in value
            if isinstance(item, (dict, list, tuple, str, int, float, bool, type(None)))
        ]
    return value


def _is_sensitive_profile_key(value: object) -> bool:
    with_boundaries = _CAMEL_CASE_BOUNDARY.sub("_", str(value).strip())
    normalized = _NON_KEY_CHARACTER.sub("_", with_boundaries.casefold()).strip("_")
    return normalized in _SENSITIVE_PROFILE_KEYS or normalized.endswith(
        _SENSITIVE_PROFILE_SUFFIXES,
    )
