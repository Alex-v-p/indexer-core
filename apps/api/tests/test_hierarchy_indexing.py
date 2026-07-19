from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from packages.indexer_application.ports import StoredDocumentFile
from packages.indexer_application.services.hierarchy_indexing import (
    build_hierarchy_summary_points,
    chunk_hierarchy_metadata,
    hierarchy_section_id,
    index_document_hierarchy,
)
from packages.rag_core.documents import DocumentChunk
from packages.rag_core.ingestion import ContextClusterSummary, DocumentContextHierarchy


class FakeEmbeddingProvider:
    vector_size = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index), 0.2, 0.3] for index, _ in enumerate(texts, start=1)]


class FakeVectorIndex:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.points = []

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1

    async def upsert_points(self, points, *, batch_size: int = 64) -> None:
        self.points.extend(points)


def _chunk(ordinal: int, *, section: str, page: int) -> DocumentChunk:
    return DocumentChunk(
        ordinal=ordinal,
        text=f"Chunk {ordinal} source text.",
        content_hash=f"hash-{ordinal}",
        token_count=5,
        source_page_start=page,
        source_page_end=page,
        section_title=section,
    )


def _stored_file(tmp_path: Path) -> StoredDocumentFile:
    path = tmp_path / "course-notes.md"
    path.write_text("source", encoding="utf-8")
    return StoredDocumentFile(
        path=path,
        storage_uri="s3://documents/course-notes.md",
        original_filename="course-notes.md",
        content_type="text/markdown",
        size_bytes=6,
        checksum_sha256="abc123",
        storage_backend="minio",
        bucket_name="documents",
        object_key="course-notes.md",
    )


def _hierarchy() -> DocumentContextHierarchy:
    return DocumentContextHierarchy(
        document_summary="Overview of retrieval strategies across the course material.",
        clusters=(
            ContextClusterSummary(
                cluster_id=1,
                chunk_ordinals=(1, 2),
                summary="Dense and keyword retrieval fundamentals.",
            ),
            ContextClusterSummary(
                cluster_id=2,
                chunk_ordinals=(3,),
                summary="Hierarchical routing from summaries to chunks.",
            ),
        ),
    )


async def test_index_document_hierarchy_reuses_document_and_cluster_summaries(tmp_path: Path) -> None:
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    chunks = [
        _chunk(1, section="Retrieval", page=2),
        _chunk(2, section="Retrieval", page=3),
        _chunk(3, section="Hierarchy", page=8),
    ]
    embeddings = FakeEmbeddingProvider()
    vector_index = FakeVectorIndex()

    count = await index_document_hierarchy(
        embedding_provider=embeddings,
        vector_index=vector_index,
        hierarchy_vector_name="hierarchy",
        document_id=document_id,
        version_id=version_id,
        version_number=4,
        uploaded_at=datetime(2026, 7, 19, tzinfo=UTC),
        published_at=datetime(2026, 6, 1, tzinfo=UTC),
        document_title="Course Notes",
        stored_file=_stored_file(tmp_path),
        chunks=chunks,
        hierarchy=_hierarchy(),
    )

    assert count == 3
    assert embeddings.calls == [[
        "Overview of retrieval strategies across the course material.",
        "Dense and keyword retrieval fundamentals.",
        "Hierarchical routing from summaries to chunks.",
    ]]
    assert vector_index.ensure_calls == 1
    assert len(vector_index.points) == 3
    assert all(set(point.vectors) == {"hierarchy"} for point in vector_index.points)
    assert all(point.payload["point_type"] == "hierarchy_summary" for point in vector_index.points)

    document_point, first_section, second_section = vector_index.points
    assert document_point.payload["hierarchy_level"] == "document"
    assert document_point.payload["document_version_number"] == 4
    assert document_point.payload["hierarchy_child_count"] == 2
    assert first_section.payload["hierarchy_level"] == "section"
    assert first_section.payload["hierarchy_section_id"] == hierarchy_section_id(version_id, 1)
    assert first_section.payload["chunk_ordinals"] == [1, 2]
    assert first_section.payload["section_title"] == "Retrieval"
    assert first_section.payload["source_page_start"] == 2
    assert first_section.payload["source_page_end"] == 3
    assert second_section.payload["hierarchy_section_id"] == hierarchy_section_id(version_id, 2)


def test_chunk_hierarchy_metadata_links_source_chunk_to_summary_scope() -> None:
    version_id = uuid.uuid4()
    chunk = _chunk(3, section="Hierarchy", page=8)

    metadata = chunk_hierarchy_metadata(
        version_id=version_id,
        chunk=chunk,
        hierarchy=_hierarchy(),
    )

    assert metadata == {
        "retrieval_level": "chunk",
        "hierarchy_document_id": str(version_id),
        "hierarchy_section_id": hierarchy_section_id(version_id, 2),
        "context_cluster_id": 2,
    }


def test_hierarchy_summary_point_ids_are_deterministic(tmp_path: Path) -> None:
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    kwargs = {
        "hierarchy_vector_name": "hierarchy",
        "document_id": document_id,
        "version_id": version_id,
        "version_number": 1,
        "uploaded_at": datetime(2026, 7, 19, tzinfo=UTC),
        "published_at": None,
        "document_title": "Course Notes",
        "stored_file": _stored_file(tmp_path),
        "chunks": [
            _chunk(1, section="Retrieval", page=2),
            _chunk(2, section="Retrieval", page=3),
            _chunk(3, section="Hierarchy", page=8),
        ],
        "hierarchy": _hierarchy(),
        "embeddings": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
    }

    first = build_hierarchy_summary_points(**kwargs)
    second = build_hierarchy_summary_points(**kwargs)

    assert [point.id for point in first] == [point.id for point in second]
