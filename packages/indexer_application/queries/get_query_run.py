from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import QueryExecutionResult
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class GetQueryRunQuery:
    query_run_id: uuid.UUID


class GetQueryRunHandler:
    def __init__(self, *, uow: UnitOfWork) -> None:
        self._uow = uow

    async def __call__(self, query: GetQueryRunQuery) -> QueryExecutionResult | None:
        query_run = await self._uow.query_runs.get(query.query_run_id)
        if query_run is None:
            return None
        return QueryExecutionResult.from_record(query_run)
