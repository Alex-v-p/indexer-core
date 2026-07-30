from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import DocumentRecord
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class GetDocumentQuery:
    document_id: uuid.UUID


class GetDocumentHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: GetDocumentQuery) -> DocumentRecord | None:
        return await self._uow.documents.get(query.document_id)
