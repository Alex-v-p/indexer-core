from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from packages.indexer_application.dto import (
    ContentGroupRecord,
    DocumentContentGroupAssignmentRecord,
    DocumentTypeDecisionRecord,
    DocumentTypeRecord,
)
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class ListDocumentTypesQuery:
    include_archived: bool = False
    limit: int = 100
    offset: int = 0


class ListDocumentTypesHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: ListDocumentTypesQuery) -> list[DocumentTypeRecord]:
        return await self._uow.document_types.list(
            include_archived=query.include_archived,
            limit=query.limit,
            offset=query.offset,
        )


@dataclass(frozen=True, slots=True)
class GetDocumentTypeQuery:
    document_type_id: uuid.UUID
    include_archived: bool = False


class GetDocumentTypeHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: GetDocumentTypeQuery) -> DocumentTypeRecord | None:
        return await self._uow.document_types.get(
            query.document_type_id,
            include_archived=query.include_archived,
        )


@dataclass(frozen=True, slots=True)
class ListContentGroupsQuery:
    include_archived: bool = False
    limit: int = 100
    offset: int = 0


class ListContentGroupsHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: ListContentGroupsQuery) -> list[ContentGroupRecord]:
        return await self._uow.content_groups.list(
            include_archived=query.include_archived,
            limit=query.limit,
            offset=query.offset,
        )


@dataclass(frozen=True, slots=True)
class GetContentGroupQuery:
    content_group_id: uuid.UUID
    include_archived: bool = False


class GetContentGroupHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: GetContentGroupQuery) -> ContentGroupRecord | None:
        return await self._uow.content_groups.get(
            query.content_group_id,
            include_archived=query.include_archived,
        )


@dataclass(frozen=True, slots=True)
class DocumentTypeDecisionView:
    decision: DocumentTypeDecisionRecord
    document_type: DocumentTypeRecord


@dataclass(frozen=True, slots=True)
class DocumentOrganizationView:
    document_id: uuid.UUID
    type_decisions: tuple[DocumentTypeDecisionView, ...]
    content_group_assignment: DocumentContentGroupAssignmentRecord | None
    content_group: ContentGroupRecord | None
    status: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class GetDocumentOrganizationQuery:
    document_id: uuid.UUID


class GetDocumentOrganizationHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: GetDocumentOrganizationQuery) -> DocumentOrganizationView:
        document = await self._uow.documents.get(query.document_id)
        if document is None:
            raise LookupError(f"Document {query.document_id} was not found.")
        decisions = await self._uow.document_types.list_document_decisions(
            document_id=query.document_id,
        )
        decision_views: list[DocumentTypeDecisionView] = []
        for decision in decisions:
            document_type = await self._uow.document_types.get(
                decision.document_type_id,
                include_archived=True,
            )
            if document_type is not None:
                decision_views.append(DocumentTypeDecisionView(decision, document_type))
        assignment = await self._uow.content_groups.get_assignment(query.document_id)
        group = None
        if assignment is not None and assignment.content_group_id is not None:
            group = await self._uow.content_groups.get(
                assignment.content_group_id,
                include_archived=True,
            )
        raw_status = document.metadata.get("organization_classification")
        organization_status = (
            {
                key: raw_status[key]
                for key in (
                    "status",
                    "job_id",
                    "document_version_id",
                    "policy_version",
                    "classifier_version",
                    "assigned_type_count",
                    "content_group_state",
                    "error_message",
                )
                if key in raw_status
            }
            if isinstance(raw_status, dict)
            else None
        )
        return DocumentOrganizationView(
            document_id=document.id,
            type_decisions=tuple(decision_views),
            content_group_assignment=assignment,
            content_group=group,
            status=organization_status,
        )
