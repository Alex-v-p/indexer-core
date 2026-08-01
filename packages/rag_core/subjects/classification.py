from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from enum import StrEnum
from math import prod
from typing import Mapping

from packages.rag_core.subjects.models import ConfidenceBand, DecisionState, SubjectKind
from packages.rag_core.subjects.naming import normalize_subject_name

CLASSIFIER_VERSION = "subject-signal-classifier/1.0"
POLICY_VERSION = "subject-decision-policy/1.0"


class SignalFamily(StrEnum):
    """Independent evidence origins used by the decision policy.

    Signals are independent only when they originate from different source
    families. Multiple canonical/alias matches inside one title, filename,
    metadata collection, or model summary are correlated and count once.
    """

    TITLE = "title"
    FILENAME = "filename"
    EXPLICIT_METADATA = "explicit_metadata"
    MODEL_SUMMARY = "model_summary"


@dataclass(frozen=True, slots=True)
class SubjectClassificationCandidate:
    subject_id: uuid.UUID
    kind: SubjectKind
    canonical_name: str
    aliases: tuple[str, ...] = ()

    @property
    def normalized_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                normalized
                for value in (self.canonical_name, *self.aliases)
                if (normalized := normalize_subject_name(value))
            ),
        )


@dataclass(frozen=True, slots=True)
class SubjectClassificationInput:
    title: str
    filename: str | None = None
    explicit_metadata_values: tuple[str, ...] = ()
    document_summary: str | None = None
    model_scores: Mapping[uuid.UUID, float] | None = None


@dataclass(frozen=True, slots=True)
class SubjectClassificationPolicy:
    high_threshold: float = 0.85
    medium_threshold: float = 0.60
    high_margin: float = 0.15
    medium_margin: float = 0.20
    minimum_suggestion_score: float = 0.25
    policy_version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        values = (
            self.high_threshold,
            self.medium_threshold,
            self.high_margin,
            self.medium_margin,
            self.minimum_suggestion_score,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Subject classification thresholds must be between 0 and 1.")
        if self.medium_threshold > self.high_threshold:
            raise ValueError("The medium threshold cannot exceed the high threshold.")
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be empty.")


@dataclass(frozen=True, slots=True)
class ClassificationSignal:
    family: SignalFamily
    label: str
    evidence_hash: str
    strength: float


@dataclass(frozen=True, slots=True)
class SubjectClassificationOutcome:
    subject_id: uuid.UUID
    state: DecisionState
    confidence: float
    confidence_band: ConfidenceBand
    margin: float
    signals: tuple[ClassificationSignal, ...]

    @property
    def independent_family_count(self) -> int:
        return len({signal.family for signal in self.signals})


def classify_document_subjects(
    request: SubjectClassificationInput,
    candidates: tuple[SubjectClassificationCandidate, ...],
    *,
    policy: SubjectClassificationPolicy | None = None,
) -> tuple[SubjectClassificationOutcome, ...]:
    """Classify against an existing active-subject catalogue without mutation."""

    resolved_policy = policy or SubjectClassificationPolicy()
    scored: list[tuple[SubjectClassificationCandidate, float, tuple[ClassificationSignal, ...]]] = []
    for candidate in candidates:
        signals = _signals_for(request, candidate)
        if not signals:
            continue
        score = round(1.0 - prod(1.0 - signal.strength for signal in signals), 6)
        if score >= resolved_policy.minimum_suggestion_score:
            scored.append((candidate, score, signals))

    scored.sort(
        key=lambda item: (item[0].kind.value, -item[1], str(item[0].subject_id)),
    )
    if not scored:
        return ()

    outcomes: list[SubjectClassificationOutcome] = []
    for kind in SubjectKind:
        group = [item for item in scored if item[0].kind is kind]
        if not group:
            continue
        lead_score = group[0][1]
        runner_score = group[1][1] if len(group) > 1 else 0.0
        lead_margin = round(lead_score - runner_score, 6)
        for position, (candidate, score, signals) in enumerate(group):
            band = _confidence_band(score, resolved_policy)
            independent_count = len({signal.family for signal in signals})
            assigned = position == 0 and (
                (
                    band is ConfidenceBand.HIGH
                    and lead_margin >= resolved_policy.high_margin
                )
                or (
                    band is ConfidenceBand.MEDIUM
                    and independent_count >= 2
                    and lead_margin >= resolved_policy.medium_margin
                )
            )
            outcomes.append(
                SubjectClassificationOutcome(
                    subject_id=candidate.subject_id,
                    state=(
                        DecisionState.ASSIGNED
                        if assigned
                        else DecisionState.SUGGESTED
                    ),
                    confidence=score,
                    confidence_band=band,
                    margin=(
                        lead_margin
                        if position == 0
                        else round(score - lead_score, 6)
                    ),
                    signals=signals,
                ),
            )
    return tuple(outcomes)


def _signals_for(
    request: SubjectClassificationInput,
    candidate: SubjectClassificationCandidate,
) -> tuple[ClassificationSignal, ...]:
    names = candidate.normalized_names
    if not names:
        return ()
    signals: list[ClassificationSignal] = []
    title_match = _matching_name(request.title, names)
    if title_match is not None:
        matched_name, exact = title_match
        signals.append(
            _signal(
                SignalFamily.TITLE,
                "exact_normalized_name" if exact else "normalized_name_phrase",
                request.title,
                matched_name,
                0.93 if exact else 0.55,
            ),
        )
    filename_match = _matching_name(request.filename or "", names)
    if filename_match is not None:
        matched_name, exact = filename_match
        signals.append(
            _signal(
                SignalFamily.FILENAME,
                "exact_normalized_name" if exact else "normalized_name_phrase",
                request.filename or "",
                matched_name,
                0.90 if exact else 0.50,
            ),
        )
    metadata_match = next(
        (
            (value, matched)
            for value in request.explicit_metadata_values
            if (matched := _matching_name(value, names)) is not None
        ),
        None,
    )
    if metadata_match is not None:
        signals.append(
            _signal(
                SignalFamily.EXPLICIT_METADATA,
                (
                    "exact_normalized_name"
                    if metadata_match[1][1]
                    else "normalized_name_phrase"
                ),
                metadata_match[0],
                metadata_match[1][0],
                0.95 if metadata_match[1][1] else 0.65,
            ),
        )
    model_score = (request.model_scores or {}).get(candidate.subject_id)
    if model_score is not None and 0.0 < model_score <= 1.0:
        signals.append(
            _signal(
                SignalFamily.MODEL_SUMMARY,
                "structured_model_score",
                request.document_summary or "",
                str(candidate.subject_id),
                float(model_score),
            ),
        )
    return tuple(signals)


def _matching_name(
    value: str,
    names: tuple[str, ...],
) -> tuple[str, bool] | None:
    normalized_value = normalize_subject_name(value)
    if not normalized_value:
        return None
    padded = f" {normalized_value} "
    return next(
        (
            (name, normalized_value == name)
            for name in names
            if f" {name} " in padded
        ),
        None,
    )


def _signal(
    family: SignalFamily,
    label: str,
    source_value: str,
    matched_value: str,
    strength: float,
) -> ClassificationSignal:
    digest = hashlib.sha256(
        f"{normalize_subject_name(source_value)}\0{matched_value}".encode("utf-8"),
    ).hexdigest()[:24]
    return ClassificationSignal(
        family=family,
        label=f"{family.value}:{label}",
        evidence_hash=digest,
        strength=round(strength, 6),
    )


def _confidence_band(
    confidence: float,
    policy: SubjectClassificationPolicy,
) -> ConfidenceBand:
    if confidence >= policy.high_threshold:
        return ConfidenceBand.HIGH
    if confidence >= policy.medium_threshold:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW
