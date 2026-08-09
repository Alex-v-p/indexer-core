from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from packages.rag_core.document_organization import (
    ClassificationConfidenceBand,
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentTypeDecisionState,
)


class CreateDocumentTypeRequest(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateDocumentTypeRequest(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    metadata: dict[str, Any] | None = None
    archive: bool = False


class DocumentTypeResponse(BaseModel):
    id: uuid.UUID
    key: str
    label: str
    description: str | None
    metadata: dict[str, Any]
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime


class CreateContentGroupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateContentGroupRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    archive: bool = False


class AddContentGroupAliasRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ContentGroupAliasResponse(BaseModel):
    id: uuid.UUID
    content_group_id: uuid.UUID
    name: str
    normalized_name: str
    archived_at: datetime | None
    created_at: datetime


class ContentGroupResponse(BaseModel):
    id: uuid.UUID
    name: str
    normalized_name: str
    description: str | None
    metadata: dict[str, Any]
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime
    aliases: list[ContentGroupAliasResponse] = Field(default_factory=list)


class ManualDocumentTypeDecisionRequest(BaseModel):
    document_type_id: uuid.UUID
    state: Literal["assigned", "rejected"]
    expected_revision: int = Field(ge=0)
    rationale: str | None = Field(default=None, max_length=2000)


class ReplaceDocumentTypesRequest(BaseModel):
    decisions: list[ManualDocumentTypeDecisionRequest]


class SetDocumentContentGroupRequest(BaseModel):
    content_group_id: uuid.UUID | None
    expected_revision: int = Field(ge=0)
    rationale: str | None = Field(default=None, max_length=2000)


class DocumentTypeDecisionResponse(BaseModel):
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


class DocumentTypeDecisionViewResponse(BaseModel):
    decision: DocumentTypeDecisionResponse
    document_type: DocumentTypeResponse


class DocumentContentGroupAssignmentResponse(BaseModel):
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


class DocumentOrganizationResponse(BaseModel):
    document_id: uuid.UUID
    type_decisions: list[DocumentTypeDecisionViewResponse]
    content_group_assignment: DocumentContentGroupAssignmentResponse | None
    content_group: ContentGroupResponse | None
    status: dict[str, Any] | None


class ReplaceDocumentTypesResponse(BaseModel):
    decisions: list[DocumentTypeDecisionResponse]


class DocumentOrganizationConflictResponse(BaseModel):
    message: str
    current_type_decisions: list[DocumentTypeDecisionResponse] = Field(default_factory=list)
    current_assignment: DocumentContentGroupAssignmentResponse | None = None


class DocumentOrganizationConflictErrorResponse(BaseModel):
    detail: DocumentOrganizationConflictResponse


class OrganizationEnqueueResponse(BaseModel):
    job_ids: list[uuid.UUID]
    skipped_document_ids: list[uuid.UUID]
