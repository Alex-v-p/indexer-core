from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Mapping

from packages.rag_core.document_organization.models import (
    ClassificationConfidenceBand,
    DocumentTypeDecisionState,
)

ORGANIZATION_CLASSIFIER_VERSION = "document-organization-classifier/1.0"
ORGANIZATION_POLICY_VERSION = "document-organization-policy/1.0"


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
    return tuple(outcomes)
