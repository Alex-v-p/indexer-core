from __future__ import annotations

import uuid
from typing import Any, Protocol

from packages.indexer_application.dto import ChunkIndexCreate, DocumentRecord, QueryRunRecord
from packages.indexer_application.ports.object_storage import StoredDocumentFile
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.agents.runtime import TraceEvent


class DocumentRepository(Protocol):
    async def create_processing_document(self, *, stored_file: StoredDocumentFile, title: str) -> uuid.UUID: ...

    async def create_processing_version(
        self,
        *,
        document_id: uuid.UUID,
        stored_file: StoredDocumentFile,
    ) -> uuid.UUID: ...

    async def set_version_parser_metadata(
        self,
        *,
        version_id: uuid.UUID,
        parser_name: str,
        parser_version: str,
        metadata: dict[str, Any],
    ) -> None: ...

    async def add_chunk_indexes(self, chunks: list[ChunkIndexCreate]) -> None: ...

    async def mark_ready(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        document_metadata: dict[str, Any],
    ) -> None: ...

    async def mark_failed(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        error_message: str,
    ) -> None: ...

    async def get(self, document_id: uuid.UUID) -> DocumentRecord | None: ...

    async def list(self, *, limit: int, offset: int) -> list[DocumentRecord]: ...


class QueryRunRepository(Protocol):
    async def create_running(
        self,
        *,
        question: str,
        pipeline_name: str,
        pipeline_version: str,
        top_k: int,
        requested_pipeline_name: str | None,
    ) -> uuid.UUID: ...

    async def mark_failed(
        self,
        *,
        query_run_id: uuid.UUID,
        error_message: str,
        trace: list[TraceEvent],
    ) -> None: ...

    async def mark_succeeded(self, *, query_run_id: uuid.UUID, state: QueryState) -> None: ...

    async def get(self, query_run_id: uuid.UUID) -> QueryRunRecord | None: ...
