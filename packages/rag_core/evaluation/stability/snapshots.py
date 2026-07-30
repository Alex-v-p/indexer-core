from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.evaluation.snapshots import citation_snapshot, evidence_snapshot
from packages.rag_core.evaluation.stability.models import (
    StabilityAttemptSnapshot,
    StructuredDiagnosticSnapshot,
)
from packages.rag_core.retrieval.models import EvidenceItem

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


def snapshot_stability_attempt(
    state: QueryState,
    *,
    attempt_number: int,
    status: str,
    top_k: int,
    duration_ms: int,
    error_type: str | None,
) -> StabilityAttemptSnapshot:
    presentation = (
        state.answer_presentation.to_metadata()
        if state.answer_presentation is not None
        else None
    )
    return StabilityAttemptSnapshot(
        attempt_number=attempt_number,
        status=status,
        top_k=top_k,
        answer=state.answer,
        answer_presentation=presentation,
        outcome=outcome_for_state(state) if status == "succeeded" else None,
        route_signature=canonical_route_signature(state),
        evidence_identities=tuple(
            sorted({stable_evidence_identity(item) for item in state.retrieved_evidence}),
        ),
        evidence=tuple(evidence_snapshot(item) for item in state.retrieved_evidence),
        citations=tuple(citation_snapshot(item) for item in state.citations),
        structured_diagnostics=safe_structured_diagnostics(_diagnostic_metadata(state)),
        runtime_profile=safe_runtime_profile(state),
        duration_ms=duration_ms,
        error_type=error_type,
    )


def _diagnostic_metadata(state: QueryState) -> dict[str, object]:
    metadata: dict[str, object] = dict(state.metadata)
    typed_reports = (
        ("query_classification", state.query_classification),
        ("information_need_decomposition", state.information_need_decomposition),
        ("evidence_grading", state.evidence_grading),
        ("retrieval_retry", state.retrieval_retry),
        ("information_need_resolution", state.information_need_resolution),
    )
    for key, value in typed_reports:
        if value is not None and key not in metadata:
            metadata[key] = value.to_metadata()
    return metadata


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
