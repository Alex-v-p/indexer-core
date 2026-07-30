from __future__ import annotations

from dataclasses import dataclass

from packages.indexer_application.dto import DocumentIngestionConfig
from packages.indexer_application.ports import MaterializedDocumentFile
from packages.indexer_application.services.ingestion.errors import IngestionError
from packages.rag_core.documents import (
    ChunkingConfig,
    DocumentChunk,
    ParsedDocument,
    chunk_document,
    parse_document,
)
from packages.rag_core.ports import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class ParseDocumentInput:
    materialized_document: MaterializedDocumentFile
    config: DocumentIngestionConfig


@dataclass(frozen=True, slots=True)
class ParsedDocumentContent:
    parsed_document: ParsedDocument
    chunks: list[DocumentChunk]
    original_embeddings: list[list[float]]


async def parse_document_content(
    *,
    request: ParseDocumentInput,
    embedding_provider: EmbeddingProvider,
) -> ParsedDocumentContent:
    """Parse, chunk, and embed the original source representation."""

    materialized = request.materialized_document
    parsed_document = parse_document(
        materialized.path,
        filename=materialized.original_filename,
        content_type=materialized.content_type,
    )
    chunks = chunk_document(
        parsed_document,
        config=ChunkingConfig(
            max_chars=request.config.chunk_max_chars,
            overlap_chars=request.config.chunk_overlap_chars,
        ),
    )
    if not chunks:
        raise IngestionError("The uploaded document did not contain any extractable text.")

    original_embeddings = await embedding_provider.embed_texts([chunk.text for chunk in chunks])
    if len(original_embeddings) != len(chunks):
        raise IngestionError("Embedding provider must return exactly one vector per source chunk.")

    return ParsedDocumentContent(
        parsed_document=parsed_document,
        chunks=chunks,
        original_embeddings=original_embeddings,
    )
