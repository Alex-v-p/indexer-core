from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from packages.rag_core.subjects.models import (
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
)

MAX_CLASSIFICATION_SIGNALS = 32
MAX_CLASSIFICATION_SIGNALS_BYTES = 16_384
MAX_SIGNAL_NAME_LENGTH = 64
MAX_RATIONALE_LENGTH = 2_000
MAX_VERSION_LENGTH = 128


@dataclass(frozen=True, slots=True)
class DocumentSubjectDecision:
    """A proposed current decision for one logical document and one subject.

    Manual rejection is deliberately represented as a durable decision rather
    than row deletion. Automatic decisions carry enough bounded provenance to
    audit and safely rerun classification independently from ingestion.
    """

    document_id: uuid.UUID
    subject_id: uuid.UUID
    state: DecisionState
    control_source: DecisionControlSource
    confidence: float | None = None
    confidence_band: ConfidenceBand | None = None
    rationale: str | None = None
    classifier_version: str | None = None
    policy_version: str | None = None
    signals: Mapping[str, Any] = field(default_factory=dict)
    classified_document_version_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", DecisionState(self.state))
        object.__setattr__(
            self,
            "control_source",
            DecisionControlSource(self.control_source),
        )
        if self.confidence_band is not None:
            object.__setattr__(
                self,
                "confidence_band",
                ConfidenceBand(self.confidence_band),
            )

        rationale = _optional_clean_text(
            self.rationale,
            field_name="rationale",
            max_length=MAX_RATIONALE_LENGTH,
        )
        classifier_version = _optional_clean_text(
            self.classifier_version,
            field_name="classifier_version",
            max_length=MAX_VERSION_LENGTH,
        )
        policy_version = _optional_clean_text(
            self.policy_version,
            field_name="policy_version",
            max_length=MAX_VERSION_LENGTH,
        )
        signals = _validated_signals(self.signals)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "classifier_version", classifier_version)
        object.__setattr__(self, "policy_version", policy_version)
        object.__setattr__(self, "signals", signals)

        if self.control_source is DecisionControlSource.AUTOMATIC:
            self._validate_automatic_provenance()
        else:
            self._validate_manual_provenance()

    @property
    def is_membership(self) -> bool:
        return self.state is DecisionState.ASSIGNED

    def _validate_automatic_provenance(self) -> None:
        if self.confidence is None or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("automatic confidence must be between 0 and 1.")
        if self.confidence_band is None:
            raise ValueError("automatic decisions require confidence_band.")
        if self.rationale is None:
            raise ValueError("automatic decisions require rationale.")
        if self.classifier_version is None:
            raise ValueError("automatic decisions require classifier_version.")
        if self.policy_version is None:
            raise ValueError("automatic decisions require policy_version.")
        if self.classified_document_version_id is None:
            raise ValueError(
                "automatic decisions require classified_document_version_id."
            )

    def _validate_manual_provenance(self) -> None:
        if self.confidence is not None or self.confidence_band is not None:
            raise ValueError("manual decisions must not carry confidence.")
        if self.classifier_version is not None or self.policy_version is not None:
            raise ValueError("manual decisions must not carry classifier provenance.")
        if self.classified_document_version_id is not None:
            raise ValueError("manual decisions must not carry a classified version.")
        if self.signals:
            raise ValueError("manual decisions must not carry classification signals.")


def _optional_clean_text(
    value: str | None,
    *,
    field_name: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError(f"{field_name} must not be blank.")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name} must not exceed {max_length} characters.")
    return cleaned


def _validated_signals(signals: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(signals)
    if len(copied) > MAX_CLASSIFICATION_SIGNALS:
        raise ValueError(
            "classification signals must not contain more than "
            f"{MAX_CLASSIFICATION_SIGNALS} entries."
        )
    if any(
        not isinstance(name, str)
        or not name.strip()
        or len(name) > MAX_SIGNAL_NAME_LENGTH
        for name in copied
    ):
        raise ValueError(
            "classification signal names must be non-empty strings no longer than "
            f"{MAX_SIGNAL_NAME_LENGTH} characters."
        )
    try:
        encoded = json.dumps(
            copied,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("classification signals must be JSON serializable.") from exc
    if len(encoded) > MAX_CLASSIFICATION_SIGNALS_BYTES:
        raise ValueError(
            "classification signals must not exceed "
            f"{MAX_CLASSIFICATION_SIGNALS_BYTES} UTF-8 JSON bytes."
        )
    return copied
