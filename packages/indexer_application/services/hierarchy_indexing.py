from __future__ import annotations

import uuid
from datetime import datetime

from packages.indexer_application.ports import StoredDocumentReference
from packages.rag_core.documents import DocumentChunk
from packages.rag_core.documents.naming import normalize_document_name
from packages.rag_core.ingestion import DocumentContextHierarchy
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter, VectorPoint

HIERARCHY_POINT_TYPE = "hierarchy_summary"
DOCUMENT_HIERARCHY_LEVEL = "document"
SECTION_HIERARCHY_LEVEL = "section"


def hierarchy_section_id(document_version_id: uuid.UUID, cluster_id: int) -> str:
    """Return the stable scope key shared by a section summary and its chunks."""

    if cluster_id <= 0:
        raise ValueError("cluster_id must be positive.")
    return f"{document_version_id}:section:{cluster_id}"


def hierarchy_document_point_id(document_version_id: uuid.UUID) -> str:
    return str(uuid.uuid5(document_version_id, "hierarchy:document"))


def hierarchy_section_point_id(document_version_id: uuid.UUID, cluster_id: int) -> str:
    if cluster_id <= 0:
        raise ValueError("cluster_id must be positive.")
    return str(uuid.uuid5(document_version_id, f"hierarchy:section:{cluster_id}"))


async def index_document_hierarchy(
    *,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndexWriter,
    hierarchy_vector_name: str,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    version_number: int,
    uploaded_at: datetime,
    published_at: datetime | None,
    document_title: str,
    stored_file: StoredDocumentReference,
    chunks: list[DocumentChunk],
    hierarchy: DocumentContextHierarchy,
) -> int:
    """Index document and semantic-section summaries as retrieval routing nodes."""

    if not hierarchy_vector_name.strip():
        raise ValueError("hierarchy_vector_name must not be empty.")
    summaries = [hierarchy.document_summary, *(cluster.summary for cluster in hierarchy.clusters)]
    if not summaries or any(not summary.strip() for summary in summaries):
        raise ValueError("Hierarchy summaries must not be empty.")

    embeddings = await embedding_provider.embed_texts(summaries)
    if len(embeddings) != len(summaries):
        raise ValueError("Embedding provider must return one vector per hierarchy summary.")

    points = build_hierarchy_summary_points(
        hierarchy_vector_name=hierarchy_vector_name,
        document_id=document_id,
        version_id=version_id,
        version_number=version_number,
        uploaded_at=uploaded_at,
        published_at=published_at,
        document_title=document_title,
        stored_file=stored_file,
        chunks=chunks,
        hierarchy=hierarchy,
        embeddings=embeddings,
    )
    await vector_index.ensure_collection()
    await vector_index.upsert_points(points)
    return len(points)


def build_hierarchy_summary_points(
    *,
    hierarchy_vector_name: str,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    version_number: int,
    uploaded_at: datetime,
    published_at: datetime | None,
    document_title: str,
    stored_file: StoredDocumentReference,
    chunks: list[DocumentChunk],
    hierarchy: DocumentContextHierarchy,
    embeddings: list[list[float]],
) -> list[VectorPoint]:
    if len(embeddings) != len(hierarchy.clusters) + 1:
        raise ValueError("Hierarchy embedding count must equal document plus section summary count.")

    common = _common_payload(
        document_id=document_id,
        version_id=version_id,
        version_number=version_number,
        uploaded_at=uploaded_at,
        published_at=published_at,
        document_title=document_title,
        stored_file=stored_file,
        hierarchy_vector_name=hierarchy_vector_name,
    )
    document_point = VectorPoint(
        id=hierarchy_document_point_id(version_id),
        vectors={hierarchy_vector_name: embeddings[0]},
        payload={
            **common,
            "text": hierarchy.document_summary,
            "summary_text": hierarchy.document_summary,
            "hierarchy_level": DOCUMENT_HIERARCHY_LEVEL,
            "hierarchy_document_id": str(version_id),
            "hierarchy_child_count": len(hierarchy.clusters),
        },
    )

    chunks_by_ordinal = {chunk.ordinal: chunk for chunk in chunks}
    section_points: list[VectorPoint] = []
    for position, cluster in enumerate(hierarchy.clusters, start=1):
        cluster_chunks = [
            chunks_by_ordinal[ordinal]
            for ordinal in cluster.chunk_ordinals
            if ordinal in chunks_by_ordinal
        ]
        if len(cluster_chunks) != len(cluster.chunk_ordinals):
            raise ValueError(f"Hierarchy cluster {cluster.cluster_id} references an unknown chunk ordinal.")
        section_titles = tuple(
            dict.fromkeys(
                chunk.section_title.strip()
                for chunk in cluster_chunks
                if chunk.section_title and chunk.section_title.strip()
            ),
        )
        page_starts = [chunk.source_page_start for chunk in cluster_chunks if chunk.source_page_start is not None]
        page_ends = [chunk.source_page_end for chunk in cluster_chunks if chunk.source_page_end is not None]
        section_scope_id = hierarchy_section_id(version_id, cluster.cluster_id)
        section_points.append(
            VectorPoint(
                id=hierarchy_section_point_id(version_id, cluster.cluster_id),
                vectors={hierarchy_vector_name: embeddings[position]},
                payload={
                    **common,
                    "text": cluster.summary,
                    "summary_text": cluster.summary,
                    "hierarchy_level": SECTION_HIERARCHY_LEVEL,
                    "hierarchy_document_id": str(version_id),
                    "hierarchy_section_id": section_scope_id,
                    "context_cluster_id": cluster.cluster_id,
                    "chunk_ordinals": list(cluster.chunk_ordinals),
                    "section_titles": list(section_titles),
                    "section_title": section_titles[0] if len(section_titles) == 1 else None,
                    "source_page_start": min(page_starts) if page_starts else None,
                    "source_page_end": max(page_ends) if page_ends else None,
                    "hierarchy_child_count": len(cluster.chunk_ordinals),
                },
            ),
        )
    return [document_point, *section_points]


def chunk_hierarchy_metadata(
    *,
    version_id: uuid.UUID,
    chunk: DocumentChunk,
    hierarchy: DocumentContextHierarchy | None,
) -> dict[str, object]:
    """Attach stable section routing metadata to an ordinary source chunk."""

    if hierarchy is None:
        return {"retrieval_level": "chunk"}
    cluster = hierarchy.cluster_for_ordinal(chunk.ordinal)
    return {
        "retrieval_level": "chunk",
        "hierarchy_document_id": str(version_id),
        "hierarchy_section_id": hierarchy_section_id(version_id, cluster.cluster_id),
        "context_cluster_id": cluster.cluster_id,
    }


def _common_payload(
    *,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    version_number: int,
    uploaded_at: datetime,
    published_at: datetime | None,
    document_title: str,
    stored_file: StoredDocumentReference,
    hierarchy_vector_name: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "point_type": HIERARCHY_POINT_TYPE,
        "document_id": str(document_id),
        "document_version_id": str(version_id),
        "document_version_number": version_number,
        "document_version_label": f"v{version_number}",
        "is_latest_version": True,
        "uploaded_at": uploaded_at.isoformat(),
        "uploaded_at_epoch": uploaded_at.timestamp(),
        "document_title": document_title,
        "document_title_normalized": normalize_document_name(document_title),
        "original_filename": stored_file.original_filename,
        "original_filename_normalized": normalize_document_name(stored_file.original_filename),
        "storage_uri": stored_file.storage_uri,
        "storage_backend": stored_file.storage_backend,
        "bucket_name": stored_file.bucket_name,
        "object_key": stored_file.object_key,
        "qdrant_vector_names": [hierarchy_vector_name],
    }
    if published_at is not None:
        payload["published_at"] = published_at.isoformat()
        payload["published_at_epoch"] = published_at.timestamp()
    return payload
