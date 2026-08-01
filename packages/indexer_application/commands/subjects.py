from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import (
    DocumentSubjectDecisionRecord,
    SubjectAliasRecord,
    SubjectRecord,
)
from packages.indexer_application.ports import (
    SubjectDecisionConflictError,
    SubjectCanonicalNameConflictError,
    UnitOfWork,
)
from packages.rag_core.subjects import (
    DecisionControlSource,
    DecisionState,
    DocumentSubjectDecision,
    SubjectKind,
    SubjectNameMatchType,
)


@dataclass(frozen=True, slots=True)
class CreateSubjectCommand:
    kind: SubjectKind
    name: str
    description: str | None = None
    metadata: dict[str, object] | None = None


class CreateSubjectHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: CreateSubjectCommand) -> SubjectRecord:
        await _ensure_canonical_name_available(
            self._uow,
            kind=command.kind,
            name=command.name,
        )
        subject = await self._uow.subjects.create(
            kind=command.kind,
            name=command.name,
            description=command.description,
            metadata=dict(command.metadata or {}),
        )
        await self._uow.commit()
        return subject


@dataclass(frozen=True, slots=True)
class UpdateSubjectCommand:
    subject_id: uuid.UUID
    name: str | None = None
    archive: bool = False


class UpdateSubjectHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: UpdateSubjectCommand) -> SubjectRecord:
        if command.name is None and not command.archive:
            raise ValueError("At least one subject update must be provided.")
        current = await self._uow.subjects.get(
            command.subject_id,
            include_archived=True,
        )
        if current is None:
            raise LookupError(f"Subject {command.subject_id} was not found.")
        if current.archived_at is not None:
            if command.name is not None:
                raise ArchivedSubjectMutationError(
                    "Archived subjects are read-only."
                )
            return current

        updated = current
        if command.name is not None:
            await _ensure_canonical_name_available(
                self._uow,
                kind=current.kind,
                name=command.name,
                excluding_subject_id=current.id,
            )
            updated = await self._uow.subjects.rename(
                subject_id=command.subject_id,
                name=command.name,
            )
        if command.archive:
            updated = await self._uow.subjects.archive(subject_id=command.subject_id)
        await self._uow.commit()
        return updated


@dataclass(frozen=True, slots=True)
class AddSubjectAliasCommand:
    subject_id: uuid.UUID
    name: str


class AddSubjectAliasHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, command: AddSubjectAliasCommand) -> SubjectAliasRecord:
        await _require_active_subject(self._uow, command.subject_id)
        alias = await self._uow.subjects.add_alias(
            subject_id=command.subject_id,
            name=command.name,
        )
        await self._uow.commit()
        return alias


@dataclass(frozen=True, slots=True)
class ArchiveSubjectAliasCommand:
    subject_id: uuid.UUID
    alias_id: uuid.UUID


class ArchiveSubjectAliasHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(
        self,
        command: ArchiveSubjectAliasCommand,
    ) -> SubjectAliasRecord:
        await _require_active_subject(self._uow, command.subject_id)
        alias = await self._uow.subjects.archive_alias(
            subject_id=command.subject_id,
            alias_id=command.alias_id,
        )
        await self._uow.commit()
        return alias


@dataclass(frozen=True, slots=True)
class SetDocumentSubjectDecisionCommand:
    document_id: uuid.UUID
    subject_id: uuid.UUID
    state: DecisionState
    expected_revision: int
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewDocumentSubjectSuggestionCommand:
    document_id: uuid.UUID
    subject_id: uuid.UUID
    accept: bool
    expected_revision: int
    rationale: str | None = None


class DocumentSubjectDecisionWriteConflict(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        current: DocumentSubjectDecisionRecord | None,
    ) -> None:
        super().__init__(message)
        self.current = current


class ArchivedSubjectMutationError(RuntimeError):
    """A mutation targeted a subject that has already been archived."""


class SetDocumentSubjectDecisionHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(
        self,
        command: SetDocumentSubjectDecisionCommand,
    ) -> DocumentSubjectDecisionRecord:
        if command.state not in (DecisionState.ASSIGNED, DecisionState.REJECTED):
            raise ValueError("Manual decisions must be assigned or rejected.")
        await _require_document(self._uow, command.document_id)
        await _require_active_subject(self._uow, command.subject_id)
        return await _write_manual_decision(
            self._uow,
            document_id=command.document_id,
            subject_id=command.subject_id,
            state=command.state,
            expected_revision=command.expected_revision,
            rationale=command.rationale,
        )


class ReviewDocumentSubjectSuggestionHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(
        self,
        command: ReviewDocumentSubjectSuggestionCommand,
    ) -> DocumentSubjectDecisionRecord:
        await _require_document(self._uow, command.document_id)
        await _require_active_subject(self._uow, command.subject_id)
        current = await self._uow.subjects.get_decision(
            document_id=command.document_id,
            subject_id=command.subject_id,
        )
        if current is None:
            raise LookupError("Document-subject suggestion was not found.")
        if (
            current.state is not DecisionState.SUGGESTED
            or current.control_source is not DecisionControlSource.AUTOMATIC
        ):
            raise ValueError("Only an automatic suggestion can be reviewed.")
        if current.revision != command.expected_revision:
            raise DocumentSubjectDecisionWriteConflict(
                "Document-subject suggestion changed before it was reviewed.",
                current=current,
            )
        return await _write_manual_decision(
            self._uow,
            document_id=command.document_id,
            subject_id=command.subject_id,
            state=(
                DecisionState.ASSIGNED
                if command.accept
                else DecisionState.REJECTED
            ),
            expected_revision=command.expected_revision,
            rationale=command.rationale,
        )


async def _write_manual_decision(
    uow: UnitOfWork,
    *,
    document_id: uuid.UUID,
    subject_id: uuid.UUID,
    state: DecisionState,
    expected_revision: int,
    rationale: str | None,
) -> DocumentSubjectDecisionRecord:
    try:
        result = await uow.subjects.write_decision(
            DocumentSubjectDecision(
                document_id=document_id,
                subject_id=subject_id,
                state=state,
                control_source=DecisionControlSource.MANUAL,
                rationale=rationale,
            ),
            expected_revision=expected_revision,
        )
    except SubjectDecisionConflictError as exc:
        current = await uow.subjects.get_decision(
            document_id=document_id,
            subject_id=subject_id,
        )
        raise DocumentSubjectDecisionWriteConflict(
            "Document-subject decision changed before this update.",
            current=current,
        ) from exc
    await uow.commit()
    return result


async def _require_document(uow: UnitOfWork, document_id: uuid.UUID) -> None:
    if await uow.documents.get(document_id) is None:
        raise LookupError(f"Document {document_id} was not found.")


async def _require_active_subject(uow: UnitOfWork, subject_id: uuid.UUID) -> SubjectRecord:
    subject = await uow.subjects.get(subject_id, include_archived=True)
    if subject is None:
        raise LookupError(f"Subject {subject_id} was not found.")
    if subject.archived_at is not None:
        raise ArchivedSubjectMutationError("Archived subjects are read-only.")
    return subject


async def _ensure_canonical_name_available(
    uow: UnitOfWork,
    *,
    kind: SubjectKind,
    name: str,
    excluding_subject_id: uuid.UUID | None = None,
) -> None:
    matches = await uow.subjects.resolve_name(
        name,
        kind=kind,
        include_archived=True,
    )
    duplicate = next(
        (
            match
            for match in matches
            if match.match_type is SubjectNameMatchType.CANONICAL
            and match.subject.id != excluding_subject_id
        ),
        None,
    )
    if duplicate is not None:
        raise SubjectCanonicalNameConflictError(
            f"A {kind.value} subject with that canonical name already exists."
        )
