from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from packages.indexer_application.dto import (
    DocumentIngestionConfig,
    DocumentRecord,
    DocumentStatus,
    DocumentVersionIdentity,
    QueryRunRecord,
    QueryRunStatus,
)
from packages.indexer_application.ports import MaterializedDocumentFile, StoredDocumentFile
from packages.indexer_application.services import run_query
from packages.indexer_application.services.background_jobs import (
    ProcessDocumentIngestionJobHandler,
    prepared_document_to_payload,
)
from packages.indexer_application.services.ingestion.prepare import (
    PrepareDocumentInput,
    prepare_document,
)
from packages.rag_core.agents import QueryState
from packages.rag_core.ingestion import (
    ContextClusterSummary,
    ContextualizationResult,
    ContextualizedChunk,
    DocumentContextHierarchy,
)


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

    async def save_upload(self, upload: FakeUpload):
        return self.stored_file.reference

    async def materialize(self, reference):
        return MaterializedDocumentFile(reference=reference, path=self.stored_file.path)

    def cleanup_materialized_file(self, materialized: MaterializedDocumentFile) -> None:
        self.cleaned = True


class FakeDocumentRepository:
    def __init__(self) -> None:
        self.document_id = uuid.uuid4()
        self.version_id = uuid.uuid4()
        self.version_candidate: DocumentRecord | None = None
        self.next_version_number = 1
        self.create_document_calls = 0
        self.version_document_id: uuid.UUID | None = None
        self.chunk_indexes = []
        self.ready_metadata = None
        self.parser_metadata = None

    async def create_processing_document(self, *, stored_file, title: str) -> uuid.UUID:
        self.create_document_calls += 1
        self.title = title
        return self.document_id

    async def create_processing_version(
        self,
        *,
        document_id,
        stored_file,
        published_at=None,
    ) -> DocumentVersionIdentity:
        self.version_document_id = document_id
        self.published_at = published_at
        return DocumentVersionIdentity(
            id=self.version_id,
            version_number=self.next_version_number,
            uploaded_at=datetime.now(UTC),
            published_at=published_at,
        )

    async def find_version_candidate(self, *, title: str, original_filename: str) -> DocumentRecord | None:
        return self.version_candidate

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
            title=getattr(self, "title", "notes"),
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

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorIndex:
    def __init__(self) -> None:
        self.points = []
        self.ensure_calls = 0
        self.events: list[tuple[str, int | str]] = []

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1

    async def upsert_points(self, points, *, batch_size: int = 64) -> None:
        self.points.extend(points)
        self.events.append(("upsert", len(points)))

    async def mark_document_version_current(self, *, document_id: str, version_id: str) -> None:
        del document_id
        self.events.append(("promote", version_id))

    async def activate_document_version(self, *, document_id, version_id) -> None:
        await self.mark_document_version_current(
            document_id=str(document_id),
            version_id=str(version_id),
        )




class FakeContextualizer:
    def __init__(self) -> None:
        self.received_embeddings = None

    async def contextualize(self, parsed_document, chunks, chunk_embeddings):
        self.received_embeddings = chunk_embeddings
        cluster = ContextClusterSummary(
            cluster_id=1,
            chunk_ordinals=tuple(chunk.ordinal for chunk in chunks),
            summary="A shared semantic cluster summary.",
        )
        return ContextualizationResult(
            chunks=[
                ContextualizedChunk(
                    chunk=chunk,
                    context=f"Context for chunk {chunk.ordinal} in {parsed_document.title}.",
                    contextualized_text=(
                        f"Context for chunk {chunk.ordinal} in {parsed_document.title}.\n\n{chunk.text}"
                    ),
                    context_cluster_id=cluster.cluster_id,
                )
                for chunk in chunks
            ],
            hierarchy=DocumentContextHierarchy(
                document_summary=f"Summary of {parsed_document.title}.",
                clusters=(cluster,),
            ),
        )


class FakeHierarchyBuilder:
    def __init__(self) -> None:
        self.calls = 0
        self.hierarchy = None

    async def build(self, parsed_document, chunks, chunk_embeddings):
        del parsed_document, chunk_embeddings
        self.calls += 1
        self.hierarchy = DocumentContextHierarchy(
            document_summary="Shared hierarchy document summary.",
            clusters=(
                ContextClusterSummary(
                    cluster_id=1,
                    chunk_ordinals=tuple(chunk.ordinal for chunk in chunks),
                    summary="Shared hierarchy semantic section summary.",
                ),
            ),
        )
        return self.hierarchy


class HierarchyAwareContextualizer:
    def __init__(self) -> None:
        self.received_hierarchy = None

    async def contextualize(self, parsed_document, chunks, chunk_embeddings, *, hierarchy=None):
        del parsed_document, chunk_embeddings
        self.received_hierarchy = hierarchy
        return ContextualizationResult(
            chunks=[
                ContextualizedChunk(
                    chunk=chunk,
                    context=f"Shared context for chunk {chunk.ordinal}.",
                    contextualized_text=f"Shared context for chunk {chunk.ordinal}.\n\n{chunk.text}",
                    context_cluster_id=1,
                )
                for chunk in chunks
            ],
            hierarchy=hierarchy,
        )


class FailingContextualizer:
    async def contextualize(self, parsed_document, chunks, chunk_embeddings):
        raise RuntimeError("context model unavailable")


class FakeCacheInvalidator:
    def __init__(self) -> None:
        self.calls = 0

    def invalidate(self) -> None:
        self.calls += 1


async def process_uploaded_document_with_worker(
    *,
    uow,
    config,
    upload,
    object_store,
    embedding_provider,
    vector_index,
    keyword_cache,
    contextualizer=None,
    hierarchy_builder=None,
    title=None,
    version_of_document_id=None,
    detect_existing_versions=True,
    published_at=None,
):
    stored_document = await object_store.save_upload(upload)
    prepared = await prepare_document(
        uow=uow,
        request=PrepareDocumentInput(
            stored_document=stored_document,
            title=title,
            version_of_document_id=version_of_document_id,
            detect_existing_versions=detect_existing_versions,
            published_at=published_at,
        ),
    )

    async def report(progress: float, stage: str) -> None:
        del progress, stage

    await ProcessDocumentIngestionJobHandler(
        uow=uow,
        config=config,
        object_store=object_store,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        version_index=vector_index,
        keyword_cache=keyword_cache,
        contextualizer=contextualizer,
        hierarchy_builder=hierarchy_builder,
    )(prepared_document_to_payload(prepared), report)
    await uow.commit()
    document = await uow.documents.get(prepared.document_id)
    assert document is not None
    return document


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

    document = await process_uploaded_document_with_worker(
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
        contextualizer=FakeContextualizer(),
    )

    assert document.status is DocumentStatus.READY
    assert uow.documents.title == "notes"
    assert len(uow.documents.chunk_indexes) == len(vector_index.points) > 1
    assert uow.flush_calls == 1
    assert uow.commit_calls == 1
    assert cache.calls == 1
    assert object_store.cleaned is True


async def test_document_ingestion_detects_matching_upload_as_next_version(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\nThe second version changes retry behavior. " * 20, encoding="utf-8")
    uow = FakeUnitOfWork()
    existing_document_id = uuid.uuid4()
    now = datetime.now(UTC)
    uow.documents.version_candidate = DocumentRecord(
        id=existing_document_id,
        title="notes",
        original_filename="notes.md",
        content_type="text/markdown",
        storage_uri="file://notes-v1.md",
        size_bytes=80,
        checksum_sha256="old",
        status=DocumentStatus.READY,
        metadata={"latest_version_number": 1},
        created_at=now,
        updated_at=now,
    )
    uow.documents.next_version_number = 2
    vector_index = FakeVectorIndex()

    await process_uploaded_document_with_worker(
        uow=uow,
        config=DocumentIngestionConfig(
            chunk_max_chars=250,
            chunk_overlap_chars=25,
            vector_collection_name="chunks",
        ),
        upload=FakeUpload(),
        object_store=FakeObjectStore(source),
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=vector_index,
        keyword_cache=FakeCacheInvalidator(),
        contextualizer=FakeContextualizer(),
    )

    assert uow.documents.create_document_calls == 0
    assert uow.documents.version_document_id == existing_document_id
    assert uow.documents.ready_metadata["version_number"] == 2
    assert uow.documents.ready_metadata["document_metadata"]["version_detection_method"] == (
        "matching_title_or_filename"
    )
    assert all(point.payload["document_id"] == str(existing_document_id) for point in vector_index.points)
    assert all(point.payload["document_version_number"] == 2 for point in vector_index.points)
    assert all(point.payload["document_version_label"] == "v2" for point in vector_index.points)


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


async def test_document_ingestion_indexes_named_original_and_contextual_vectors_on_one_point(tmp_path: Path) -> None:
    source = tmp_path / "contextual.md"
    source.write_text("# Context\n\nThe system stores original evidence separately. " * 20, encoding="utf-8")
    uow = FakeUnitOfWork()
    vector_index = FakeVectorIndex()
    embeddings = FakeEmbeddingProvider()
    cache = FakeCacheInvalidator()
    contextualizer = FakeContextualizer()

    await process_uploaded_document_with_worker(
        uow=uow,
        config=DocumentIngestionConfig(
            chunk_max_chars=250,
            chunk_overlap_chars=25,
            vector_collection_name="chunks",
            original_vector_name="original",
            contextual_vector_name="contextual",
        ),
        upload=FakeUpload(),
        object_store=FakeObjectStore(source),
        embedding_provider=embeddings,
        vector_index=vector_index,
        contextualizer=contextualizer,
        keyword_cache=cache,
    )

    assert len(vector_index.points) > 1
    assert len(embeddings.calls) == 2
    assert contextualizer.received_embeddings is not None
    assert len(contextualizer.received_embeddings) == len(vector_index.points)
    assert embeddings.calls[0] == [point.payload["text"] for point in vector_index.points]
    assert all(text.startswith("Context for chunk") for text in embeddings.calls[1])
    first_point = vector_index.points[0]
    assert set(first_point.vectors) == {"original", "contextual"}
    assert first_point.payload["text"] in first_point.payload["contextualized_text"]
    assert first_point.payload["contextualized_text"].startswith("Context for chunk 1")
    assert first_point.payload["contextualization_status"] == "ready"
    assert uow.documents.chunk_indexes[0].qdrant_collection == "chunks"
    assert uow.documents.chunk_indexes[0].metadata["qdrant_vector_names"] == ["original", "contextual"]
    contextualization = uow.documents.ready_metadata["document_metadata"]["contextualization"]
    assert contextualization["status"] == "ready"
    assert contextualization["strategy"] == "semantic_cluster_hierarchy"
    assert contextualization["cluster_count"] == 1
    assert contextualization["document_summary"] == "Summary of contextual."
    assert contextualization["collection"] == "chunks"
    assert contextualization["vector_name"] == "contextual"
    assert first_point.payload["context_cluster_id"] == 1
    assert cache.calls == 1


async def test_document_ingestion_reuses_hierarchy_for_contextualization_and_routing_points(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hierarchical.md"
    source.write_text("# Hierarchy\n\nBroad routing should narrow to precise chunks. " * 20, encoding="utf-8")
    uow = FakeUnitOfWork()
    vector_index = FakeVectorIndex()
    embeddings = FakeEmbeddingProvider()
    hierarchy_builder = FakeHierarchyBuilder()
    contextualizer = HierarchyAwareContextualizer()

    await process_uploaded_document_with_worker(
        uow=uow,
        config=DocumentIngestionConfig(
            chunk_max_chars=250,
            chunk_overlap_chars=25,
            vector_collection_name="chunks",
            original_vector_name="original",
            contextual_vector_name="contextual",
            hierarchy_vector_name="hierarchy",
            contextualization_enabled=True,
            hierarchical_indexing_enabled=True,
        ),
        upload=FakeUpload(),
        object_store=FakeObjectStore(source),
        embedding_provider=embeddings,
        vector_index=vector_index,
        contextualizer=contextualizer,
        hierarchy_builder=hierarchy_builder,
        keyword_cache=FakeCacheInvalidator(),
    )

    assert hierarchy_builder.calls == 1
    assert contextualizer.received_hierarchy is hierarchy_builder.hierarchy
    chunk_points = [point for point in vector_index.points if point.payload.get("retrieval_level") == "chunk"]
    summary_points = [point for point in vector_index.points if point.payload.get("point_type") == "hierarchy_summary"]
    assert chunk_points
    assert len(summary_points) == 2
    assert all("hierarchy_section_id" in point.payload for point in chunk_points)
    assert {point.payload["hierarchy_level"] for point in summary_points} == {"document", "section"}
    assert set(summary_points[0].vectors) == {"hierarchy"}
    assert vector_index.events[-1][0] == "promote"
    hierarchical_metadata = uow.documents.ready_metadata["document_metadata"]["hierarchical_retrieval"]
    assert hierarchical_metadata["status"] == "ready"
    assert hierarchical_metadata["summary_point_count"] == 2


async def test_document_ingestion_can_fail_open_when_contextualization_fails(tmp_path: Path) -> None:
    source = tmp_path / "fallback.md"
    source.write_text("# Fallback\n\nOriginal indexing should still complete. " * 20, encoding="utf-8")
    uow = FakeUnitOfWork()
    vector_index = FakeVectorIndex()

    document = await process_uploaded_document_with_worker(
        uow=uow,
        config=DocumentIngestionConfig(
            chunk_max_chars=250,
            chunk_overlap_chars=25,
            vector_collection_name="chunks",
            original_vector_name="original",
            contextual_vector_name="contextual",
            contextualization_enabled=True,
            contextualization_fail_open=True,
        ),
        upload=FakeUpload(),
        object_store=FakeObjectStore(source),
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=vector_index,
        contextualizer=FailingContextualizer(),
        keyword_cache=FakeCacheInvalidator(),
    )

    assert document.status is DocumentStatus.READY
    assert vector_index.points
    assert all(set(point.vectors) == {"original"} for point in vector_index.points)
    assert all("contextualized_text" not in point.payload for point in vector_index.points)
    assert all(point.payload["contextualization_status"] == "failed_open" for point in vector_index.points)
    assert all(
        point.payload["contextualization_error"] == "context model unavailable"
        for point in vector_index.points
    )
    contextualization = uow.documents.ready_metadata["document_metadata"]["contextualization"]
    assert contextualization["status"] == "failed_open"
    assert contextualization["error"] == "context model unavailable"
    assert contextualization["collection"] == "chunks"
    assert contextualization["vector_name"] == "contextual"



async def test_document_ingestion_detects_draft_suffix_as_version_family(tmp_path: Path) -> None:
    source = tmp_path / "Realization_Draft5.md"
    source.write_text("# Realization\n\nVersion five content. " * 30, encoding="utf-8")
    uow = FakeUnitOfWork()
    existing_document_id = uuid.uuid4()
    now = datetime.now(UTC)
    uow.documents.version_candidate = DocumentRecord(
        id=existing_document_id,
        title="Realization_Draft4",
        original_filename="Realization_Draft4.md",
        content_type="text/markdown",
        storage_uri="file://Realization_Draft4.md",
        size_bytes=80,
        checksum_sha256="old",
        status=DocumentStatus.READY,
        metadata={"latest_version_number": 1},
        created_at=now,
        updated_at=now,
    )
    uow.documents.next_version_number = 2

    await process_uploaded_document_with_worker(
        uow=uow,
        config=DocumentIngestionConfig(
            chunk_max_chars=250,
            chunk_overlap_chars=25,
            vector_collection_name="chunks",
        ),
        upload=FakeUpload(),
        object_store=FakeObjectStore(source),
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex(),
        keyword_cache=FakeCacheInvalidator(),
        contextualizer=FakeContextualizer(),
        published_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert uow.documents.create_document_calls == 0
    assert uow.documents.version_document_id == existing_document_id
    assert uow.documents.parser_metadata["metadata"]["version_detection"]["method"] == "matching_document_family"
    assert uow.documents.published_at == datetime(2026, 6, 1, tzinfo=UTC)
    assert all(
        chunk.metadata["published_at"].startswith("2026-06-01")
        for chunk in uow.documents.chunk_indexes
    )
