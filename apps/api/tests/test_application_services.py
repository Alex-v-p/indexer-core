from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from packages.indexer_application.dto import (
    DocumentIngestionConfig,
    DocumentRecord,
    DocumentStatus,
    QueryRunRecord,
    QueryRunStatus,
)
from packages.indexer_application.ports import StoredDocumentFile
from packages.indexer_application.services import ingest_uploaded_document, run_query
from packages.rag_core.agents import QueryState


class FakeUpload:
    filename = "notes.md"
    content_type = "text/markdown"

    async def read(self, size: int = -1) -> bytes:
        return b""


class FakeObjectStore:
    def __init__(self, path: Path) -> None:
        self.stored_file = StoredDocumentFile(
            path=path,
            storage_uri=f"file://{path}",
            original_filename=path.name,
            content_type="text/markdown",
            size_bytes=path.stat().st_size,
            checksum_sha256="abc",
            storage_backend="local",
        )
        self.cleaned = False

    async def save_upload(self, upload: FakeUpload) -> StoredDocumentFile:
        return self.stored_file

    def cleanup_staging_file(self, stored_file: StoredDocumentFile) -> None:
        self.cleaned = True


class FakeDocumentRepository:
    def __init__(self) -> None:
        self.document_id = uuid.uuid4()
        self.version_id = uuid.uuid4()
        self.chunk_indexes = []
        self.ready_metadata = None
        self.parser_metadata = None

    async def create_processing_document(self, *, stored_file, title: str) -> uuid.UUID:
        self.title = title
        return self.document_id

    async def create_processing_version(self, *, document_id, stored_file) -> uuid.UUID:
        return self.version_id

    async def set_version_parser_metadata(self, **kwargs) -> None:
        self.parser_metadata = kwargs

    async def add_chunk_indexes(self, chunks) -> None:
        self.chunk_indexes.extend(chunks)

    async def mark_ready(self, **kwargs) -> None:
        self.ready_metadata = kwargs

    async def mark_failed(self, **kwargs) -> None:
        raise AssertionError("The successful ingestion path must not mark the document failed.")

    async def get(self, document_id: uuid.UUID) -> DocumentRecord:
        now = datetime.now(UTC)
        return DocumentRecord(
            id=document_id,
            title=self.title,
            original_filename="notes.md",
            content_type="text/markdown",
            storage_uri="file://notes.md",
            size_bytes=100,
            checksum_sha256="abc",
            status=DocumentStatus.READY,
            metadata={"chunk_count": len(self.chunk_indexes)},
            created_at=now,
            updated_at=now,
        )

    async def list(self, *, limit: int, offset: int):
        return []


class FakeQueryRunRepository:
    def __init__(self) -> None:
        self.query_run_id = uuid.uuid4()
        self.record = None

    async def create_running(self, **kwargs) -> uuid.UUID:
        self.created = kwargs
        return self.query_run_id

    async def mark_succeeded(self, *, query_run_id: uuid.UUID, state: QueryState) -> None:
        now = datetime.now(UTC)
        self.record = QueryRunRecord(
            id=query_run_id,
            question=state.question,
            answer=state.answer,
            status=QueryRunStatus.SUCCEEDED,
            pipeline_name=state.pipeline_name,
            pipeline_version=state.pipeline_version,
            top_k=state.top_k,
            started_at=now,
            completed_at=now,
            error_message=None,
            metadata=state.metadata,
        )

    async def mark_failed(self, **kwargs) -> None:
        raise AssertionError("The successful query path must not be marked failed.")

    async def get(self, query_run_id: uuid.UUID) -> QueryRunRecord | None:
        return self.record


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.documents = FakeDocumentRepository()
        self.query_runs = FakeQueryRunRepository()
        self.flush_calls = 0
        self.commit_calls = 0

    async def flush(self) -> None:
        self.flush_calls += 1

    async def commit(self) -> None:
        self.commit_calls += 1


class FakeEmbeddingProvider:
    vector_size = 3

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorIndex:
    def __init__(self) -> None:
        self.points = []
        self.ensure_calls = 0

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1

    async def upsert_points(self, points, *, batch_size: int = 64) -> None:
        self.points.extend(points)


class FakeCacheInvalidator:
    def __init__(self) -> None:
        self.calls = 0

    def invalidate(self) -> None:
        self.calls += 1


class StubPipeline:
    name = "stub"
    version = "1.0.0"

    async def run(self, state: QueryState) -> QueryState:
        state.answer = "Grounded answer."
        return state


async def test_document_ingestion_service_uses_ports_without_api_dependencies(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\nHybrid retrieval combines vector and keyword evidence. " * 20, encoding="utf-8")
    uow = FakeUnitOfWork()
    object_store = FakeObjectStore(source)
    vector_index = FakeVectorIndex()
    cache = FakeCacheInvalidator()

    document = await ingest_uploaded_document(
        uow=uow,
        config=DocumentIngestionConfig(
            chunk_max_chars=250,
            chunk_overlap_chars=25,
            vector_collection_name="chunks",
        ),
        upload=FakeUpload(),
        object_store=object_store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=vector_index,
        keyword_cache=cache,
    )

    assert document.status is DocumentStatus.READY
    assert uow.documents.title == "notes"
    assert len(uow.documents.chunk_indexes) == len(vector_index.points) > 1
    assert uow.flush_calls == 1
    assert uow.commit_calls == 1
    assert cache.calls == 1
    assert object_store.cleaned is True


async def test_query_service_receives_selected_pipeline_and_persists_result() -> None:
    uow = FakeUnitOfWork()

    result = await run_query(
        uow=uow,
        pipeline=StubPipeline(),
        question="How does the graph work?",
        top_k=4,
        requested_pipeline_name="stub",
    )

    assert result.answer == "Grounded answer."
    assert result.pipeline_name == "stub"
    assert uow.query_runs.created["requested_pipeline_name"] == "stub"
    assert uow.commit_calls == 1
