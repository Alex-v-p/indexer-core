from __future__ import annotations

import uuid

from packages.indexer_application.dto import ChunkIndexCreate, DocumentIngestionConfig
from packages.indexer_application.ports import CacheInvalidator, StoredDocumentFile, UnitOfWork
from packages.rag_core.documents.models import DocumentChunk
from packages.rag_core.ingestion import ContextualizedChunk
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
    contextualized_chunks: list[ContextualizedChunk] | None = None,
    contextualization_metadata: dict[str, object] | None = None,
) -> None:
    """Index one Qdrant point per source chunk with named representations."""

    contextual_by_ordinal = _contextual_chunks_by_ordinal(chunks, contextualized_chunks)

    await vector_index.ensure_collection()
    original_embeddings = await embedding_provider.embed_texts([chunk.text for chunk in chunks])

    contextual_embeddings: list[list[float]] = []
    if contextual_by_ordinal:
        contextual_embeddings = await embedding_provider.embed_texts(
            [contextual_by_ordinal[chunk.ordinal].contextualized_text for chunk in chunks],
        )

    index_records: list[ChunkIndexCreate] = []
    points: list[VectorPoint] = []
    for position, (chunk, original_embedding) in enumerate(zip(chunks, original_embeddings, strict=True)):
        chunk_index_id = uuid.uuid4()
        point_id = str(uuid.uuid4())
        contextualized = contextual_by_ordinal.get(chunk.ordinal)
        vector_names = [config.original_vector_name]
        vectors = {config.original_vector_name: original_embedding}
        if contextualized is not None:
            vector_names.append(config.contextual_vector_name)
            vectors[config.contextual_vector_name] = contextual_embeddings[position]

        metadata = build_chunk_metadata(
            chunk=chunk,
            document_id=document_id,
            version_id=version_id,
            stored_file=stored_file,
            chunk_index_id=chunk_index_id,
            vector_names=vector_names,
            contextualized_chunk=contextualized,
            contextualization_metadata=contextualization_metadata,
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
                vectors=vectors,
                payload=_build_point_payload(
                    chunk=chunk,
                    metadata=metadata,
                    contextualized_chunk=contextualized,
                ),
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
    vector_names: list[str],
    contextualized_chunk: ContextualizedChunk | None = None,
    contextualization_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        **chunk.metadata,
        "document_id": str(document_id),
        "document_version_id": str(version_id),
        "qdrant_chunk_index_id": str(chunk_index_id),
        "original_filename": stored_file.original_filename,
        "storage_uri": stored_file.storage_uri,
        "storage_backend": stored_file.storage_backend,
        "bucket_name": stored_file.bucket_name,
        "object_key": stored_file.object_key,
        "ordinal": chunk.ordinal,
        "qdrant_vector_names": vector_names,
    }
    if contextualized_chunk is not None:
        metadata.update(
            {
                "contextualization_status": "ready",
                "contextual_context": contextualized_chunk.context,
            },
        )
    else:
        contextualization = contextualization_metadata or {}
        status = str(contextualization.get("status") or "not_indexed")
        metadata["contextualization_status"] = status
        error = contextualization.get("error")
        if error:
            metadata["contextualization_error"] = str(error)
    return metadata


def _build_point_payload(
    *,
    chunk: DocumentChunk,
    metadata: dict[str, object],
    contextualized_chunk: ContextualizedChunk | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "token_count": chunk.token_count,
        **metadata,
    }
    if contextualized_chunk is not None:
        payload["contextualized_text"] = contextualized_chunk.contextualized_text
    return payload


def _contextual_chunks_by_ordinal(
    chunks: list[DocumentChunk],
    contextualized_chunks: list[ContextualizedChunk] | None,
) -> dict[int, ContextualizedChunk]:
    if not contextualized_chunks:
        return {}
    if len(contextualized_chunks) != len(chunks):
        raise ValueError("Contextualized chunk count must match the original chunk count.")

    original_ordinals = {chunk.ordinal for chunk in chunks}
    contextual_by_ordinal = {item.chunk.ordinal: item for item in contextualized_chunks}
    if len(contextual_by_ordinal) != len(contextualized_chunks):
        raise ValueError("Contextualized chunks must have unique ordinals.")
    if set(contextual_by_ordinal) != original_ordinals:
        raise ValueError("Contextualized chunk ordinals must match the original chunks.")
    for chunk in chunks:
        contextualized = contextual_by_ordinal[chunk.ordinal]
        if contextualized.chunk.content_hash != chunk.content_hash:
            raise ValueError("Contextualized chunks must reference the matching original chunk content.")
    return contextual_by_ordinal
