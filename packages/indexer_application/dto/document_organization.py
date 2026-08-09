from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from packages.rag_core.document_organization import (
    ClassificationConfidenceBand,
    ClassificationSource,
    ContentGroupAssignmentState,
    ContentGroupNameMatchType,
    DocumentTypeDecisionState,
)


@dataclass(frozen=True, slots=True)
class DocumentTypeRecord:
    id: uuid.UUID
    key: str
    label: str
    description: str | None
    metadata: dict[str, Any]
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentTypeDecisionRecord:
    id: uuid.UUID
    document_id: uuid.UUID
    document_type_id: uuid.UUID
    state: DocumentTypeDecisionState
    source: ClassificationSource
    confidence: float | None
    confidence_band: ClassificationConfidenceBand | None
    rationale: str | None
    classifier_version: str | None
    policy_version: str | None
    signals: dict[str, Any]
    classified_document_version_id: uuid.UUID | None
    revision: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ContentGroupAliasRecord:
    id: uuid.UUID
    content_group_id: uuid.UUID
    name: str
    normalized_name: str
    archived_at: datetime | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ContentGroupRecord:
    id: uuid.UUID
    name: str
    normalized_name: str
    description: str | None
    metadata: dict[str, Any]
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime
    aliases: tuple[ContentGroupAliasRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ContentGroupNameMatchRecord:
    content_group: ContentGroupRecord
    match_type: ContentGroupNameMatchType


@dataclass(frozen=True, slots=True)
class DocumentContentGroupAssignmentRecord:
    document_id: uuid.UUID
    content_group_id: uuid.UUID | None
    state: ContentGroupAssignmentState
    source: ClassificationSource
    unresolved_reason: str | None
    confidence: float | None
    confidence_band: ClassificationConfidenceBand | None
    rationale: str | None
    classifier_version: str | None
    policy_version: str | None
    signals: dict[str, Any]
    summary_hash: str | None
    classified_document_version_id: uuid.UUID | None
    revision: int
    created_at: datetime
    updated_at: datetime

    @property
    def is_resolved(self) -> bool:
        return self.content_group_id is not None
