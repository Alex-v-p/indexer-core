from __future__ import annotations

import uuid

from packages.indexer_application.dto import ChunkIndexCreate, DocumentIngestionConfig
from packages.indexer_application.ports import CacheInvalidator, StoredDocumentFile, UnitOfWork
from packages.rag_core.documents.models import DocumentChunk
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter, VectorPoint


async def index_document_chunks(
    *,
    uow: UnitOfWork,
    config: DocumentIngestionConfig,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndexWriter,
    keyword_cache: CacheInvalidator,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    stored_file: StoredDocumentFile,
    chunks: list[DocumentChunk],
) -> None:
    """Embed chunks, write them to the vector index, and persist point references."""

    await vector_index.ensure_collection()
    embeddings = await embedding_provider.embed_texts([chunk.text for chunk in chunks])

    index_records: list[ChunkIndexCreate] = []
    points: list[VectorPoint] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_index_id = uuid.uuid4()
        point_id = str(uuid.uuid4())
        metadata = build_chunk_metadata(
            chunk=chunk,
            document_id=document_id,
            version_id=version_id,
            stored_file=stored_file,
            chunk_index_id=chunk_index_id,
        )
        index_records.append(
            ChunkIndexCreate(
                id=chunk_index_id,
                document_id=document_id,
                document_version_id=version_id,
                ordinal=chunk.ordinal,
                content_hash=chunk.content_hash,
                token_count=chunk.token_count,
                source_page_start=chunk.source_page_start,
                source_page_end=chunk.source_page_end,
                section_title=chunk.section_title,
                qdrant_collection=config.vector_collection_name,
                qdrant_point_id=point_id,
                metadata=metadata,
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

    await uow.documents.add_chunk_indexes(index_records)
    await uow.flush()
    await vector_index.upsert_points(points)
    keyword_cache.invalidate()


def build_chunk_metadata(
    *,
    chunk: DocumentChunk,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    stored_file: StoredDocumentFile,
    chunk_index_id: uuid.UUID,
) -> dict[str, object]:
    return {
        **chunk.metadata,
        "document_id": str(document_id),
        "document_version_id": str(version_id),
        "qdrant_chunk_index_id": str(chunk_index_id),
        "original_filename": stored_file.original_filename,
        "storage_uri": stored_file.storage_uri,
        "storage_backend": stored_file.storage_backend,
        "bucket_name": stored_file.bucket_name,
        "object_key": stored_file.object_key,
    }
