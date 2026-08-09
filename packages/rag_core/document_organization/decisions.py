from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from packages.rag_core.document_organization.models import (
    ClassificationConfidenceBand,
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentTypeDecisionState,
)

MAX_SIGNALS = 32
MAX_SIGNALS_BYTES = 16_384
MAX_RATIONALE_LENGTH = 2_000
MAX_VERSION_LENGTH = 128
MAX_UNRESOLVED_REASON_LENGTH = 1_000
MAX_SUMMARY_HASH_LENGTH = 128


@dataclass(frozen=True, slots=True)
class DocumentTypeDecision:
    document_id: uuid.UUID
    document_type_id: uuid.UUID
    state: DocumentTypeDecisionState
    source: ClassificationSource
    confidence: float | None = None
    confidence_band: ClassificationConfidenceBand | None = None
    rationale: str | None = None
    classifier_version: str | None = None
    policy_version: str | None = None
    signals: Mapping[str, Any] = field(default_factory=dict)
    classified_document_version_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", DocumentTypeDecisionState(self.state))
        object.__setattr__(self, "source", ClassificationSource(self.source))
        _normalize_provenance(self)
        if self.source is ClassificationSource.AUTOMATIC:
            _require_automatic_provenance(self)
        else:
            _require_manual_provenance(self)

    @property
    def is_membership(self) -> bool:
        return self.state is DocumentTypeDecisionState.ASSIGNED


@dataclass(frozen=True, slots=True)
class DocumentContentGroupAssignment:
    document_id: uuid.UUID
    content_group_id: uuid.UUID | None
    state: ContentGroupAssignmentState
    source: ClassificationSource
    unresolved_reason: str | None = None
    confidence: float | None = None
    confidence_band: ClassificationConfidenceBand | None = None
    rationale: str | None = None
    classifier_version: str | None = None
    policy_version: str | None = None
    signals: Mapping[str, Any] = field(default_factory=dict)
    summary_hash: str | None = None
    classified_document_version_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ContentGroupAssignmentState(self.state))
        object.__setattr__(self, "source", ClassificationSource(self.source))
        _normalize_provenance(self)
        object.__setattr__(
            self,
            "unresolved_reason",
            _optional_text(self.unresolved_reason, "unresolved_reason", MAX_UNRESOLVED_REASON_LENGTH),
        )
        object.__setattr__(
            self,
            "summary_hash",
            _optional_text(self.summary_hash, "summary_hash", MAX_SUMMARY_HASH_LENGTH),
        )
        resolved = self.state in {
            ContentGroupAssignmentState.SUGGESTED,
            ContentGroupAssignmentState.ASSIGNED,
        }
        if resolved != (self.content_group_id is not None):
            raise ValueError("resolved content group states require a group and unresolved states forbid one.")
        if self.state is ContentGroupAssignmentState.UNRESOLVED and self.unresolved_reason is None:
            raise ValueError("unresolved content group assignments require unresolved_reason.")
        if resolved and self.unresolved_reason is not None:
            raise ValueError("resolved content group assignments must not carry unresolved_reason.")
        if self.source is ClassificationSource.MANUAL:
            if self.state is not ContentGroupAssignmentState.ASSIGNED:
                raise ValueError("manual content group assignments must be assigned.")
            _require_manual_provenance(self, include_summary_hash=True)
        elif resolved:
            _require_automatic_provenance(self, include_summary_hash=True)

    @property
    def is_resolved(self) -> bool:
        return self.state in {
            ContentGroupAssignmentState.SUGGESTED,
            ContentGroupAssignmentState.ASSIGNED,
        }


def _normalize_provenance(value: object) -> None:
    band = getattr(value, "confidence_band")
    if band is not None:
        object.__setattr__(value, "confidence_band", ClassificationConfidenceBand(band))
    for name, maximum in (
        ("rationale", MAX_RATIONALE_LENGTH),
        ("classifier_version", MAX_VERSION_LENGTH),
        ("policy_version", MAX_VERSION_LENGTH),
    ):
        object.__setattr__(value, name, _optional_text(getattr(value, name), name, maximum))
    object.__setattr__(value, "signals", _validated_signals(getattr(value, "signals")))


def _require_automatic_provenance(value: object, *, include_summary_hash: bool = False) -> None:
    confidence = getattr(value, "confidence")
    if confidence is None or isinstance(confidence, bool) or not 0.0 <= confidence <= 1.0:
        raise ValueError("automatic confidence must be between 0 and 1.")
    for name in ("confidence_band", "rationale", "classifier_version", "policy_version", "classified_document_version_id"):
        if getattr(value, name) is None:
            raise ValueError(f"automatic decisions require {name}.")
    if include_summary_hash and getattr(value, "summary_hash") is None:
        raise ValueError("automatic resolved content group assignments require summary_hash.")


def _require_manual_provenance(value: object, *, include_summary_hash: bool = False) -> None:
    forbidden = (
        "confidence", "confidence_band", "classifier_version", "policy_version",
        "classified_document_version_id",
    )
    if any(getattr(value, name) is not None for name in forbidden):
        raise ValueError("manual decisions must not carry model provenance.")
    if getattr(value, "signals"):
        raise ValueError("manual decisions must not carry classification signals.")
    if include_summary_hash and getattr(value, "summary_hash") is not None:
        raise ValueError("manual assignments must not carry summary_hash.")


def _optional_text(value: str | None, name: str, maximum: int) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{name} must be non-blank and no longer than {maximum} characters.")
    return cleaned


def _validated_signals(signals: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(signals)
    if len(copied) > MAX_SIGNALS or any(
        not isinstance(key, str) or not key.strip() or len(key) > 64 for key in copied
    ):
        raise ValueError("classification signals are invalid or exceed 32 entries.")
    try:
        encoded = json.dumps(copied, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("classification signals must be JSON serializable.") from exc
    if len(encoded) > MAX_SIGNALS_BYTES:
        raise ValueError("classification signals must not exceed 16384 UTF-8 JSON bytes.")
    return copied
