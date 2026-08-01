from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from packages.rag_core.subjects import (
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
    SubjectKind,
    SubjectNameMatchType,
)


@dataclass(frozen=True, slots=True)
class SubjectAliasRecord:
    id: uuid.UUID
    subject_id: uuid.UUID
    name: str
    normalized_name: str
    created_at: datetime
    archived_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SubjectRecord:
    id: uuid.UUID
    kind: SubjectKind
    name: str
    normalized_name: str
    description: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    aliases: tuple[SubjectAliasRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class SubjectNameMatchRecord:
    subject: SubjectRecord
    match_type: SubjectNameMatchType


@dataclass(frozen=True, slots=True)
class DocumentSubjectDecisionRecord:
    id: uuid.UUID
    document_id: uuid.UUID
    subject_id: uuid.UUID
    state: DecisionState
    control_source: DecisionControlSource
    confidence: float | None
    confidence_band: ConfidenceBand | None
    rationale: str | None
    classifier_version: str | None
    policy_version: str | None
    signals: dict[str, Any]
    classified_document_version_id: uuid.UUID | None
    revision: int
    created_at: datetime
    updated_at: datetime

    @property
    def is_membership(self) -> bool:
        return self.state is DecisionState.ASSIGNED
