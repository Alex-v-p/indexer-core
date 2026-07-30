from __future__ import annotations

from dataclasses import dataclass

from packages.indexer_application.dto import DocumentRecord
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class ListDocumentsQuery:
    limit: int = 50
    offset: int = 0


class ListDocumentsHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: ListDocumentsQuery) -> list[DocumentRecord]:
        return await self._uow.documents.list(limit=query.limit, offset=query.offset)
