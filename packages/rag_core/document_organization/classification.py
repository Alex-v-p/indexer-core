from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Mapping

import re
import unicodedata

from packages.rag_core.document_organization.models import (
    ClassificationConfidenceBand,
    DocumentTypeDecisionState,
)

ORGANIZATION_CLASSIFIER_VERSION = "document-organization-classifier/1.2"
ORGANIZATION_POLICY_VERSION = "document-organization-policy/1.1"


@dataclass(frozen=True, slots=True)
class DocumentOrganizationPolicy:
    high_threshold: float = 0.85
    medium_threshold: float = 0.60
    confirmation_margin: float = 0.20
    policy_version: str = ORGANIZATION_POLICY_VERSION

    def __post_init__(self) -> None:
        if not 0 <= self.medium_threshold <= self.high_threshold <= 1:
            raise ValueError("organization confidence thresholds are invalid.")
        if not 0 <= self.confirmation_margin <= 1:
            raise ValueError("confirmation_margin must be between 0 and 1.")
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be blank.")


@dataclass(frozen=True, slots=True)
class DocumentTypeCandidate:
    id: uuid.UUID
    key: str
    label: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentTypeOutcome:
    document_type_id: uuid.UUID
    state: DocumentTypeDecisionState
    confidence: float
    confidence_band: ClassificationConfidenceBand


def select_document_types(
    scores: Mapping[str, float],
    candidates: tuple[DocumentTypeCandidate, ...],
    *,
    policy: DocumentOrganizationPolicy,
) -> tuple[DocumentTypeOutcome, ...]:
    """Select every supported medium/high type from an extensible catalogue."""

    by_key = {candidate.key: candidate for candidate in candidates}
    outcomes: list[DocumentTypeOutcome] = []
    for key, confidence in scores.items():
        candidate = by_key.get(key)
        if candidate is None or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            continue
        if confidence < policy.medium_threshold:
            continue
        outcomes.append(
            DocumentTypeOutcome(
                document_type_id=candidate.id,
                state=DocumentTypeDecisionState.ASSIGNED,
                confidence=float(confidence),
                confidence_band=(
                    ClassificationConfidenceBand.HIGH
                    if confidence >= policy.high_threshold
                    else ClassificationConfidenceBand.MEDIUM
                ),
            )
        )
    outcomes.sort(key=lambda item: (-item.confidence, str(item.document_type_id)))
    other = by_key.get("other")
    if other is not None and any(item.document_type_id != other.id for item in outcomes):
        outcomes = [item for item in outcomes if item.document_type_id != other.id]
    return tuple(outcomes)


_ROLE_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("projectplan", "project plan"), ("plan",)),
    (("realization", "realisation"), ("realization", "realisation", "report")),
    (("functionalspec", "functional spec", "functional specification"), ("specification", "spec")),
)


def merge_explicit_document_role_scores(
    model_scores: Mapping[str, float],
    *,
    title: str,
    filename: str | None,
    candidates: tuple[DocumentTypeCandidate, ...],
    explicit_confidence: float = 0.98,
) -> dict[str, float]:
    """Merge deterministic role words without inventing catalogue keys."""

    if not 0 <= explicit_confidence <= 1:
        raise ValueError("explicit_confidence must be between 0 and 1.")
    supported = {candidate.key: candidate for candidate in candidates}
    merged = {
        key: confidence
        for key, confidence in model_scores.items()
        if key in supported
    }
    source_tokens = _tokenize_role(f"{title} {filename or ''}")
    explicit_keys: set[str] = set()

    for candidate in candidates:
        if candidate.key == "other":
            continue
        forms = (_tokenize_role(candidate.key), _tokenize_role(candidate.label))
        if any(_contains_role_phrase(source_tokens, form) for form in forms if form):
            explicit_keys.add(candidate.key)

    candidate_forms: dict[str, str] = {}
    for candidate in candidates:
        if candidate.key == "other":
            continue
        for value in (candidate.key, candidate.label):
            normalized = " ".join(_tokenize_role(value))
            candidate_forms.setdefault(normalized, candidate.key)
            candidate_forms.setdefault(normalized.replace(" ", ""), candidate.key)
    for aliases, preferred_roles in _ROLE_ALIASES:
        if not any(_contains_role_phrase(source_tokens, _tokenize_role(alias)) for alias in aliases):
            continue
        target = next(
            (candidate_forms[role] for role in preferred_roles if role in candidate_forms),
            None,
        )
        if target is not None:
            explicit_keys.add(target)

    for key in explicit_keys:
        merged[key] = max(merged.get(key, 0.0), explicit_confidence)
    return merged


def _tokenize_role(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value)
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    camel_split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", camel_split)
    folded = camel_split.casefold()
    return tuple(re.sub(r"[\W_]+", " ", folded).split())


def _contains_role_phrase(source: tuple[str, ...], target: tuple[str, ...]) -> bool:
    if not target or len(target) > len(source):
        return False
    return any(
        source[index:index + len(target)] == target
        for index in range(len(source) - len(target) + 1)
    )
