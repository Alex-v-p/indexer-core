from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import (
    DocumentSubjectDecisionRecord,
    SubjectNameMatchRecord,
    SubjectRecord,
)
from packages.indexer_application.ports import UnitOfWork
from packages.rag_core.subjects import DecisionState, SubjectKind


@dataclass(frozen=True, slots=True)
class ListSubjectsQuery:
    kind: SubjectKind | None = None
    include_archived: bool = False
    limit: int = 100
    offset: int = 0


class ListSubjectsHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: ListSubjectsQuery) -> list[SubjectRecord]:
        return await self._uow.subjects.list(
            kind=query.kind,
            include_archived=query.include_archived,
            limit=query.limit,
            offset=query.offset,
        )


@dataclass(frozen=True, slots=True)
class GetSubjectQuery:
    subject_id: uuid.UUID
    include_archived: bool = False


class GetSubjectHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: GetSubjectQuery) -> SubjectRecord | None:
        return await self._uow.subjects.get(
            query.subject_id,
            include_archived=query.include_archived,
        )


@dataclass(frozen=True, slots=True)
class ResolveSubjectNameQuery:
    name: str
    kind: SubjectKind | None = None
    include_archived: bool = False


class ResolveSubjectNameHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(
        self,
        query: ResolveSubjectNameQuery,
    ) -> list[SubjectNameMatchRecord]:
        return await self._uow.subjects.resolve_name(
            query.name,
            kind=query.kind,
            include_archived=query.include_archived,
        )


@dataclass(frozen=True, slots=True)
class ListDocumentSubjectDecisionsQuery:
    document_id: uuid.UUID
    states: tuple[DecisionState, ...] | None = None


class ListDocumentSubjectDecisionsHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(
        self,
        query: ListDocumentSubjectDecisionsQuery,
    ) -> list[DocumentSubjectDecisionRecord]:
        if await self._uow.documents.get(query.document_id) is None:
            raise LookupError(f"Document {query.document_id} was not found.")
        return await self._uow.subjects.list_document_decisions(
            document_id=query.document_id,
            states=query.states,
        )


@dataclass(frozen=True, slots=True)
class ListDocumentSubjectSuggestionsQuery:
    document_id: uuid.UUID


class ListDocumentSubjectSuggestionsHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(
        self,
        query: ListDocumentSubjectSuggestionsQuery,
    ) -> list[DocumentSubjectDecisionRecord]:
        if await self._uow.documents.get(query.document_id) is None:
            raise LookupError(f"Document {query.document_id} was not found.")
        return await self._uow.subjects.list_document_suggestions(
            document_id=query.document_id,
        )
