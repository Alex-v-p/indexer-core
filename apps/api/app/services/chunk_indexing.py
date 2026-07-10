from __future__ import annotations

import uuid
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.database.models import Document, DocumentVersion, QdrantChunkIndex
from app.adapters.embeddings import build_embedding_provider
from app.adapters.object_storage import StoredDocumentFile
from app.adapters.vector_store import build_vector_store
from app.core.config import Settings
from packages.rag_core.documents.models import DocumentChunk
from packages.rag_core.providers.vector_stores import VectorPoint


class VectorStore(Protocol):
    async def ensure_collection(self) -> None:
        """Ensure the target collection exists."""

    async def upsert_points(self, points: list[VectorPoint], *, batch_size: int = 64) -> None:
        """Upsert vector points."""


async def index_document_chunks(
    *,
    session: AsyncSession,
    settings: Settings,
    document: Document,
    version: DocumentVersion,
    stored_file: StoredDocumentFile,
    chunks: list[DocumentChunk],
) -> None:
    """Embed chunks, store them in Qdrant, and record their point references.

    Postgres remains a lightweight registry for lifecycle/provenance. Chunk text
    and vectors live in the vector store payload for Phase 1 retrieval.
    """

    embedding_provider = build_embedding_provider(settings)
    vector_store = build_vector_store(settings)

    await vector_store.ensure_collection()
    embeddings = await embedding_provider.embed_texts([chunk.text for chunk in chunks])

    points: list[VectorPoint] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_index_id = uuid.uuid4()
        point_id = str(uuid.uuid4())
        metadata = build_chunk_metadata(
            chunk=chunk,
            document=document,
            version=version,
            stored_file=stored_file,
            chunk_index_id=chunk_index_id,
        )
        session.add(
            QdrantChunkIndex(
                id=chunk_index_id,
                document_id=document.id,
                document_version_id=version.id,
                ordinal=chunk.ordinal,
                content_hash=chunk.content_hash,
                token_count=chunk.token_count,
                source_page_start=chunk.source_page_start,
                source_page_end=chunk.source_page_end,
                section_title=chunk.section_title,
                qdrant_collection=settings.qdrant_collection,
                qdrant_point_id=point_id,
                metadata_=metadata,
            ),
        )
        points.append(
            VectorPoint(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "content_hash": chunk.content_hash,
                    "token_count": chunk.token_count,
                    **metadata,
                },
            ),
        )

    await session.flush()
    await vector_store.upsert_points(points)


def build_chunk_metadata(
    *,
    chunk: DocumentChunk,
    document: Document,
    version: DocumentVersion,
    stored_file: StoredDocumentFile,
    chunk_index_id: uuid.UUID,
) -> dict[str, object]:
    return {
        **chunk.metadata,
        "document_id": str(document.id),
        "document_version_id": str(version.id),
        "qdrant_chunk_index_id": str(chunk_index_id),
        "original_filename": stored_file.original_filename,
        "storage_uri": stored_file.storage_uri,
        "storage_backend": stored_file.storage_backend,
        "bucket_name": stored_file.bucket_name,
        "object_key": stored_file.object_key,
    }
