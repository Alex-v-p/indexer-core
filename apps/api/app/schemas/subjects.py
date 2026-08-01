from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from packages.rag_core.subjects import (
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
    SubjectKind,
    SubjectNameMatchType,
)


class CreateSubjectRequest(BaseModel):
    kind: SubjectKind
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateSubjectRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    archive: bool = False


class AddSubjectAliasRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class SubjectAliasResponse(BaseModel):
    id: uuid.UUID
    subject_id: uuid.UUID
    name: str
    normalized_name: str
    created_at: datetime
    archived_at: datetime | None = None


class SubjectResponse(BaseModel):
    id: uuid.UUID
    kind: SubjectKind
    name: str
    normalized_name: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    aliases: list[SubjectAliasResponse] = Field(default_factory=list)


class SubjectNameMatchResponse(BaseModel):
    subject: SubjectResponse
    match_type: SubjectNameMatchType


class SubjectNameResolutionResponse(BaseModel):
    normalized_name: str
    ambiguous: bool
    matches: list[SubjectNameMatchResponse] = Field(default_factory=list)


class SetDocumentSubjectDecisionRequest(BaseModel):
    state: Literal[DecisionState.ASSIGNED, DecisionState.REJECTED]
    expected_revision: int = Field(ge=0)
    rationale: str | None = Field(default=None, max_length=2_000)


class ReviewSubjectSuggestionRequest(BaseModel):
    decision: Literal["accept", "reject"]
    expected_revision: int = Field(ge=1)
    rationale: str | None = Field(default=None, max_length=2_000)


class DocumentSubjectDecisionResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    subject_id: uuid.UUID
    state: DecisionState
    control_source: DecisionControlSource
    confidence: float | None = None
    confidence_band: ConfidenceBand | None = None
    rationale: str | None = None
    classifier_version: str | None = None
    policy_version: str | None = None
    signals: dict[str, Any] = Field(default_factory=dict)
    classified_document_version_id: uuid.UUID | None = None
    revision: int
    created_at: datetime
    updated_at: datetime


class SubjectDecisionConflictResponse(BaseModel):
    message: str
    current: DocumentSubjectDecisionResponse | None = None


class SubjectDecisionConflictErrorResponse(BaseModel):
    detail: SubjectDecisionConflictResponse
