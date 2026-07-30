from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from packages.indexer_application.dto import DocumentIngestionConfig
from packages.indexer_application.ports import CacheInvalidator, UnitOfWork
from packages.indexer_application.services.chunk_indexing import index_document_chunks
from packages.indexer_application.services.hierarchy_indexing import index_document_hierarchy
from packages.indexer_application.services.ingestion.contextualize import ContextualizedDocumentContent
from packages.indexer_application.services.ingestion.errors import IngestionError
from packages.indexer_application.services.ingestion.parse import ParsedDocumentContent
from packages.indexer_application.services.ingestion.prepare import PreparedDocument
from packages.rag_core.ports import EmbeddingProvider, VectorIndexWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexDocumentInput:
    config: DocumentIngestionConfig
    prepared: PreparedDocument
    parsed: ParsedDocumentContent
    contextualized: ContextualizedDocumentContent


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    hierarchical_retrieval_metadata: dict[str, object]


async def index_document(
    *,
    request: IndexDocumentInput,
    uow: UnitOfWork,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndexWriter,
    keyword_cache: CacheInvalidator,
) -> IndexedDocument:
    """Write chunk and hierarchy indexes without activating the version."""

    config = request.config
    prepared = request.prepared
    parsed = request.parsed
    contextualized = request.contextualized
    version = prepared.version
    uploaded_at = version.uploaded_at or datetime.now(UTC)
    indexed_hierarchy = (
        contextualized.hierarchy if config.hierarchical_indexing_enabled else None
    )

    await index_document_chunks(
        uow=uow,
        config=config,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        keyword_cache=keyword_cache,
        document_id=prepared.document_id,
        version_id=version.id,
        version_number=version.version_number,
        uploaded_at=uploaded_at,
        published_at=version.published_at,
        document_title=prepared.document_title,
        stored_file=prepared.stored_document,
        chunks=parsed.chunks,
        original_embeddings=parsed.original_embeddings,
        contextualized_chunks=contextualized.contextualized_chunks,
        contextualization_metadata=contextualized.contextualization_metadata,
        hierarchy=indexed_hierarchy,
    )
    hierarchical_metadata = await _index_hierarchy(
        config=config,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        prepared=prepared,
        parsed=parsed,
        contextualized=contextualized,
        uploaded_at=uploaded_at,
    )
    return IndexedDocument(hierarchical_retrieval_metadata=hierarchical_metadata)


async def _index_hierarchy(
    *,
    config: DocumentIngestionConfig,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndexWriter,
    prepared: PreparedDocument,
    parsed: ParsedDocumentContent,
    contextualized: ContextualizedDocumentContent,
    uploaded_at: datetime,
) -> dict[str, object]:
    if not config.hierarchical_indexing_enabled:
        return {"enabled": False, "status": "disabled"}
    metadata: dict[str, object] = {
        **contextualized.hierarchy_build_metadata,
        "enabled": True,
        "collection": config.vector_collection_name,
        "vector_name": config.hierarchy_vector_name,
        "levels": ["document", "section", "chunk"],
    }
    hierarchy = contextualized.hierarchy
    if hierarchy is None:
        error = str(metadata.get("error") or "No document hierarchy was produced.")
        if not config.hierarchical_indexing_fail_open:
            raise IngestionError(error)
        return {**metadata, "status": "failed_open", "error": error}

    try:
        summary_point_count = await index_document_hierarchy(
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            hierarchy_vector_name=config.hierarchy_vector_name,
            document_id=prepared.document_id,
            version_id=prepared.version.id,
            version_number=prepared.version.version_number,
            uploaded_at=uploaded_at,
            published_at=prepared.version.published_at,
            document_title=prepared.document_title,
            stored_file=prepared.stored_document,
            chunks=parsed.chunks,
            hierarchy=hierarchy,
        )
    except Exception as exc:
        if not config.hierarchical_indexing_fail_open:
            logger.exception("Hierarchy summary indexing failed; aborting ingestion.")
            raise
        logger.warning(
            "Hierarchy summary indexing failed; continuing with chunk indexes only.",
            exc_info=True,
        )
        return {**metadata, "status": "failed_open", "error": str(exc)}

    return {
        **metadata,
        "status": "ready",
        "summary_point_count": summary_point_count,
        "document_summary_count": 1,
        "section_summary_count": len(hierarchy.clusters),
        "document_summary": hierarchy.document_summary,
        "sections": [
            {
                "cluster_id": cluster.cluster_id,
                "chunk_ordinals": list(cluster.chunk_ordinals),
                "summary": cluster.summary,
            }
            for cluster in hierarchy.clusters
        ],
    }
