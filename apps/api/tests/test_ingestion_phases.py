from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.indexer_application.dto import (
    DocumentIngestionConfig,
    DocumentRecord,
    DocumentStatus,
    DocumentVersionIdentity,
)
from packages.indexer_application.ports import MaterializedDocumentFile, StoredDocumentReference
from packages.indexer_application.services.ingestion import (
    ActivateDocumentInput,
    ContextualizationInput,
    ContextualizedDocumentContent,
    IndexDocumentInput,
    IndexedDocument,
    IngestionError,
    ParseDocumentInput,
    ParsedDocumentContent,
    PrepareDocumentInput,
    PreparedDocument,
    activate_document,
    contextualize_document,
    index_document,
    parse_document_content,
    prepare_document,
)
from packages.rag_core.documents import DocumentChunk, ParsedDocument, ParsedPage


def _reference(name: str = "notes.md") -> StoredDocumentReference:
    return StoredDocumentReference(
        storage_uri=f"s3://documents/{name}",
        original_filename=name,
        content_type="text/markdown",
        size_bytes=10,
        checksum_sha256="abc",
        storage_backend="minio",
        bucket_name="documents",
        object_key=name,
    )


def _config() -> DocumentIngestionConfig:
    return DocumentIngestionConfig(
        chunk_max_chars=200,
        chunk_overlap_chars=20,
        vector_collection_name="chunks",
        contextualization_enabled=False,
        hierarchical_indexing_enabled=False,
    )


class FakeDocuments:
    def __init__(self) -> None:
        self.document_id = uuid.uuid4()
        self.version = DocumentVersionIdentity(
            id=uuid.uuid4(),
            version_number=2,
            uploaded_at=datetime(2026, 7, 30, tzinfo=UTC),
        )
        self.failed: dict | None = None
        self.chunk_indexes = []
        self.ready: dict | None = None
        self.parser_metadata: dict | None = None

    async def get(self, document_id):
        return None

    async def find_version_candidate(self, *, title, original_filename):
        return None

    async def create_processing_document(self, *, stored_file, title):
        self.created = (stored_file, title)
        return self.document_id

    async def create_processing_version(self, *, document_id, stored_file, published_at=None):
        self.version_created = (document_id, stored_file, published_at)
        return self.version

    async def add_chunk_indexes(self, chunks):
        self.chunk_indexes.extend(chunks)

    async def set_version_parser_metadata(self, **kwargs):
        self.parser_metadata = kwargs

    async def mark_ready(self, **kwargs):
        self.ready = kwargs

    async def mark_failed(self, **kwargs):
        self.failed = kwargs

    async def list(self, *, limit, offset):
        return []


class FakeUow:
    def __init__(self) -> None:
        self.documents = FakeDocuments()
        self.query_runs = object()
        self.flush_calls = 0
        self.commit_calls = 0

    async def flush(self):
        self.flush_calls += 1

    async def commit(self):
        self.commit_calls += 1


class FakeEmbeddingProvider:
    vector_size = 3

    async def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorIndex:
    def __init__(self) -> None:
        self.points = []

    async def ensure_collection(self):
        return None

    async def upsert_points(self, points, *, batch_size=64):
        self.points.extend(points)

    async def mark_document_version_current(self, *, document_id, version_id):
        return None


class FakeActivator:
    def __init__(self) -> None:
        self.calls = []

    async def activate_document_version(self, *, document_id, version_id):
        self.calls.append((document_id, version_id))


class FakeCache:
    def __init__(self) -> None:
        self.calls = 0

    def invalidate(self):
        self.calls += 1


async def test_prepare_phase_creates_typed_processing_identity() -> None:
    uow = FakeUow()
    result = await prepare_document(
        uow=uow,
        request=PrepareDocumentInput(stored_document=_reference(), title="Notes"),
    )

    assert isinstance(result, PreparedDocument)
    assert result.document_id == uow.documents.document_id
    assert result.version == uow.documents.version
    assert result.document_title == "Notes"
    assert result.version_detection_method == "new_document"


async def test_parse_phase_materializes_parser_output_chunks_and_embeddings(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nTyped phases make ingestion easier to test. " * 8, encoding="utf-8")
    result = await parse_document_content(
        request=ParseDocumentInput(
            materialized_document=MaterializedDocumentFile(reference=_reference(), path=path),
            config=_config(),
        ),
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert result.parsed_document.parser_name
    assert result.chunks
    assert len(result.original_embeddings) == len(result.chunks)


async def test_contextualize_phase_reports_disabled_without_side_effects() -> None:
    parsed = ParsedDocumentContent(
        parsed_document=ParsedDocument(
            title="Notes",
            pages=[ParsedPage(page_number=1, text="content")],
            parser_name="test",
            parser_version="1",
        ),
        chunks=[DocumentChunk(ordinal=1, text="content", content_hash="hash", token_count=1)],
        original_embeddings=[[0.1, 0.2, 0.3]],
    )
    result = await contextualize_document(
        request=ContextualizationInput(config=_config(), parsed=parsed),
        contextualizer=None,
        hierarchy_builder=None,
    )

    assert result.contextualized_chunks is None
    assert result.contextualization_metadata == {"enabled": False, "status": "disabled"}
    assert result.hierarchy is None


async def test_index_phase_writes_chunks_without_activating_version() -> None:
    uow = FakeUow()
    prepared = PreparedDocument(
        stored_document=_reference(),
        document_id=uow.documents.document_id,
        document_title="Notes",
        version=uow.documents.version,
        version_detection_method="new_document",
        matched_existing_document=False,
    )
    parsed = ParsedDocumentContent(
        parsed_document=ParsedDocument(
            title="Notes",
            pages=[ParsedPage(page_number=1, text="content")],
            parser_name="test",
            parser_version="1",
        ),
        chunks=[DocumentChunk(ordinal=1, text="content", content_hash="hash", token_count=1)],
        original_embeddings=[[0.1, 0.2, 0.3]],
    )
    contextualized = ContextualizedDocumentContent(
        hierarchy=None,
        hierarchy_build_metadata={"enabled": False, "status": "disabled"},
        contextualized_chunks=None,
        contextualization_metadata={"enabled": False, "status": "disabled"},
    )
    vector_index = FakeVectorIndex()
    cache = FakeCache()

    result = await index_document(
        request=IndexDocumentInput(
            config=_config(),
            prepared=prepared,
            parsed=parsed,
            contextualized=contextualized,
        ),
        uow=uow,
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=vector_index,
        keyword_cache=cache,
    )

    assert result.hierarchical_retrieval_metadata == {"enabled": False, "status": "disabled"}
    assert len(uow.documents.chunk_indexes) == len(vector_index.points) == 1
    assert uow.flush_calls == 1
    assert cache.calls == 1


async def test_activate_phase_stages_version_promotion_without_committing() -> None:
    uow = FakeUow()
    activator = FakeActivator()
    prepared = PreparedDocument(
        stored_document=_reference(),
        document_id=uow.documents.document_id,
        document_title="Notes",
        version=uow.documents.version,
        version_detection_method="new_document",
        matched_existing_document=False,
    )
    parsed = ParsedDocumentContent(
        parsed_document=ParsedDocument(
            title="Notes",
            pages=[ParsedPage(page_number=1, text="content")],
            parser_name="test",
            parser_version="1",
        ),
        chunks=[DocumentChunk(ordinal=1, text="content", content_hash="hash", token_count=1)],
        original_embeddings=[[0.1, 0.2, 0.3]],
    )
    contextualized = ContextualizedDocumentContent(
        hierarchy=None,
        hierarchy_build_metadata={"enabled": False, "status": "disabled"},
        contextualized_chunks=None,
        contextualization_metadata={"enabled": False, "status": "disabled"},
    )

    await activate_document(
        request=ActivateDocumentInput(
            prepared=prepared,
            parsed=parsed,
            contextualized=contextualized,
            indexed=IndexedDocument(
                hierarchical_retrieval_metadata={"enabled": False, "status": "disabled"}
            ),
        ),
        uow=uow,
        version_index=activator,
    )

    assert activator.calls == [(prepared.document_id, prepared.version.id)]
    assert uow.documents.ready is not None
    assert uow.documents.parser_metadata is not None
    assert uow.commit_calls == 0


class BytesUpload:
    filename = "stored.md"
    content_type = "text/markdown"

    def __init__(self, content: bytes) -> None:
        self._content = content
        self._offset = 0

    async def read(self, size=-1):
        if self._offset >= len(self._content):
            return b""
        if size < 0:
            size = len(self._content) - self._offset
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


async def test_local_object_store_separates_stable_reference_from_materialization(tmp_path: Path) -> None:
    from packages.indexer_infrastructure.object_storage import LocalDocumentObjectStore

    store = LocalDocumentObjectStore(base_dir=tmp_path / "objects")
    reference = await store.save_upload(BytesUpload(b"# Stored\n\nDurable bytes."))

    assert isinstance(reference, StoredDocumentReference)
    assert not hasattr(reference, "path")
    materialized = await store.materialize(reference)
    assert materialized.reference is reference
    assert materialized.path.read_bytes() == b"# Stored\n\nDurable bytes."
    store.cleanup_materialized_file(materialized)
    assert materialized.path.exists()
    await store.delete(reference)
    assert not materialized.path.exists()
