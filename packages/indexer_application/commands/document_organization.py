from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import (
    ContentGroupAliasRecord,
    ContentGroupRecord,
    DocumentContentGroupAssignmentRecord,
    DocumentTypeDecisionRecord,
    DocumentTypeRecord,
)
from packages.indexer_application.ports import (
    DocumentOrganizationConflictError,
    UnitOfWork,
)
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentContentGroupAssignment,
    DocumentTypeDecision,
    DocumentTypeDecisionState,
)


class ArchivedDocumentOrganizationMutationError(RuntimeError):
    pass


class DocumentOrganizationWriteConflict(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        current_type_decisions: tuple[DocumentTypeDecisionRecord, ...] = (),
        current_assignment: DocumentContentGroupAssignmentRecord | None = None,
    ) -> None:
        super().__init__(message)
        self.current_type_decisions = current_type_decisions
        self.current_assignment = current_assignment


@dataclass(frozen=True, slots=True)
class CreateDocumentTypeCommand:
    key: str
    label: str
    description: str | None = None
    metadata: dict[str, object] | None = None


class CreateDocumentTypeHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: CreateDocumentTypeCommand) -> DocumentTypeRecord:
        result = await self._uow.document_types.create(
            key=command.key,
            label=command.label,
            description=command.description,
            metadata=dict(command.metadata or {}),
        )
        await self._uow.commit()
        return result


@dataclass(frozen=True, slots=True)
class UpdateDocumentTypeCommand:
    document_type_id: uuid.UUID
    label: str | None = None
    description: str | None = None
    metadata: dict[str, object] | None = None
    archive: bool = False


class UpdateDocumentTypeHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: UpdateDocumentTypeCommand) -> DocumentTypeRecord:
        current = await self._uow.document_types.get(command.document_type_id, include_archived=True)
        if current is None:
            raise LookupError(f"Document type {command.document_type_id} was not found.")
        if current.archived_at is not None:
            if any(value is not None for value in (command.label, command.description, command.metadata)):
                raise ArchivedDocumentOrganizationMutationError("Archived document types are read-only.")
            return current
        if not command.archive and all(
            value is None
            for value in (command.label, command.description, command.metadata)
        ):
            raise ValueError("At least one document type update must be provided.")
        result = current
        if any(value is not None for value in (command.label, command.description, command.metadata)):
            result = await self._uow.document_types.update(
                command.document_type_id,
                label=command.label,
                description=command.description,
                metadata=dict(command.metadata) if command.metadata is not None else None,
            )
        if command.archive:
            result = await self._uow.document_types.archive(command.document_type_id)
        await self._uow.commit()
        return result


@dataclass(frozen=True, slots=True)
class CreateContentGroupCommand:
    name: str
    description: str | None = None
    metadata: dict[str, object] | None = None


class CreateContentGroupHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: CreateContentGroupCommand) -> ContentGroupRecord:
        result = await self._uow.content_groups.create(
            name=command.name,
            description=command.description,
            metadata=dict(command.metadata or {}),
        )
        await self._uow.commit()
        return result


@dataclass(frozen=True, slots=True)
class UpdateContentGroupCommand:
    content_group_id: uuid.UUID
    name: str | None = None
    archive: bool = False


class UpdateContentGroupHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: UpdateContentGroupCommand) -> ContentGroupRecord:
        if command.name is None and not command.archive:
            raise ValueError("At least one content group update must be provided.")
        current = await self._uow.content_groups.get(command.content_group_id, include_archived=True)
        if current is None:
            raise LookupError(f"Content group {command.content_group_id} was not found.")
        if current.archived_at is not None:
            if command.name is not None:
                raise ArchivedDocumentOrganizationMutationError("Archived content groups are read-only.")
            return current
        result = current
        if command.name is not None:
            result = await self._uow.content_groups.rename(
                content_group_id=command.content_group_id,
                name=command.name,
            )
        if command.archive:
            result = await self._uow.content_groups.archive(command.content_group_id)
        await self._uow.commit()
        return result


@dataclass(frozen=True, slots=True)
class AddContentGroupAliasCommand:
    content_group_id: uuid.UUID
    name: str


class AddContentGroupAliasHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: AddContentGroupAliasCommand) -> ContentGroupAliasRecord:
        await _require_active_group(self._uow, command.content_group_id)
        result = await self._uow.content_groups.add_alias(
            content_group_id=command.content_group_id,
            name=command.name,
        )
        await self._uow.commit()
        return result


@dataclass(frozen=True, slots=True)
class ArchiveContentGroupAliasCommand:
    content_group_id: uuid.UUID
    alias_id: uuid.UUID


class ArchiveContentGroupAliasHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: ArchiveContentGroupAliasCommand) -> ContentGroupAliasRecord:
        await _require_active_group(self._uow, command.content_group_id)
        result = await self._uow.content_groups.archive_alias(
            content_group_id=command.content_group_id,
            alias_id=command.alias_id,
        )
        await self._uow.commit()
        return result


@dataclass(frozen=True, slots=True)
class ManualDocumentTypeDecisionInput:
    document_type_id: uuid.UUID
    state: DocumentTypeDecisionState
    expected_revision: int
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class ReplaceDocumentTypesCommand:
    document_id: uuid.UUID
    decisions: tuple[ManualDocumentTypeDecisionInput, ...]


class ReplaceDocumentTypesHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: ReplaceDocumentTypesCommand) -> list[DocumentTypeDecisionRecord]:
        await _require_document(self._uow, command.document_id)
        ids = [item.document_type_id for item in command.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("Each document type may appear only once.")
        results: list[DocumentTypeDecisionRecord] = []
        for item in command.decisions:
            if item.state not in {DocumentTypeDecisionState.ASSIGNED, DocumentTypeDecisionState.REJECTED}:
                raise ValueError("Manual document type decisions must be assigned or rejected.")
            await _require_active_type(self._uow, item.document_type_id)
            try:
                result = await self._uow.document_types.write_decision(
                    DocumentTypeDecision(
                        document_id=command.document_id,
                        document_type_id=item.document_type_id,
                        state=item.state,
                        source=ClassificationSource.MANUAL,
                        rationale=item.rationale,
                    ),
                    expected_revision=item.expected_revision,
                )
            except DocumentOrganizationConflictError as exc:
                await self._uow.rollback()
                current = await self._uow.document_types.list_document_decisions(document_id=command.document_id)
                raise DocumentOrganizationWriteConflict(
                    "Document type decisions changed before this replacement.",
                    current_type_decisions=tuple(current),
                ) from exc
            results.append(result)
        await self._uow.commit()
        return results


@dataclass(frozen=True, slots=True)
class SetDocumentContentGroupCommand:
    document_id: uuid.UUID
    content_group_id: uuid.UUID | None
    expected_revision: int
    rationale: str | None = None


class SetDocumentContentGroupHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(
        self,
        command: SetDocumentContentGroupCommand,
    ) -> DocumentContentGroupAssignmentRecord | None:
        await _require_document(self._uow, command.document_id)
        if command.content_group_id is None:
            current = await self._uow.content_groups.get_assignment(command.document_id)
            if current is None:
                if command.expected_revision != 0:
                    raise DocumentOrganizationWriteConflict(
                        "Content group assignment changed before it was cleared.",
                    )
                return None
            try:
                result = await self._uow.content_groups.clear_assignment(
                    document_id=command.document_id,
                    expected_revision=command.expected_revision,
                )
            except DocumentOrganizationConflictError as exc:
                latest = await self._uow.content_groups.get_assignment(command.document_id)
                raise DocumentOrganizationWriteConflict(
                    "Content group assignment changed before it was cleared.",
                    current_assignment=latest,
                ) from exc
        else:
            await _require_active_group(self._uow, command.content_group_id)
            try:
                result = await self._uow.content_groups.write_assignment(
                    DocumentContentGroupAssignment(
                        document_id=command.document_id,
                        content_group_id=command.content_group_id,
                        state=ContentGroupAssignmentState.ASSIGNED,
                        source=ClassificationSource.MANUAL,
                        rationale=command.rationale,
                    ),
                    expected_revision=command.expected_revision,
                )
            except DocumentOrganizationConflictError as exc:
                latest = await self._uow.content_groups.get_assignment(command.document_id)
                raise DocumentOrganizationWriteConflict(
                    "Content group assignment changed before this update.",
                    current_assignment=latest,
                ) from exc
        await self._uow.commit()
        return result


async def _require_document(uow: UnitOfWork, document_id: uuid.UUID) -> None:
    if await uow.documents.get(document_id) is None:
        raise LookupError(f"Document {document_id} was not found.")


async def _require_active_type(uow: UnitOfWork, document_type_id: uuid.UUID) -> DocumentTypeRecord:
    value = await uow.document_types.get(document_type_id, include_archived=True)
    if value is None:
        raise LookupError(f"Document type {document_type_id} was not found.")
    if value.archived_at is not None:
        raise ArchivedDocumentOrganizationMutationError("Archived document types are read-only.")
    return value


async def _require_active_group(uow: UnitOfWork, content_group_id: uuid.UUID) -> ContentGroupRecord:
    value = await uow.content_groups.get(content_group_id, include_archived=True)
    if value is None:
        raise LookupError(f"Content group {content_group_id} was not found.")
    if value.archived_at is not None:
        raise ArchivedDocumentOrganizationMutationError("Archived content groups are read-only.")
    return value
