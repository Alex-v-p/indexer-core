from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass

from packages.indexer_application.dto import DocumentIngestionConfig
from packages.indexer_application.services.ingestion.errors import IngestionError
from packages.indexer_application.services.ingestion.parse import ParsedDocumentContent
from packages.rag_core.ingestion import (
    ChunkContextualizer,
    ContextualizedChunk,
    DocumentContextHierarchy,
    DocumentContextHierarchyBuilder,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextualizationInput:
    config: DocumentIngestionConfig
    parsed: ParsedDocumentContent


@dataclass(frozen=True, slots=True)
class ContextualizedDocumentContent:
    hierarchy: DocumentContextHierarchy | None
    hierarchy_build_metadata: dict[str, object]
    contextualized_chunks: list[ContextualizedChunk] | None
    contextualization_metadata: dict[str, object]


async def contextualize_document(
    *,
    request: ContextualizationInput,
    contextualizer: ChunkContextualizer | None,
    hierarchy_builder: DocumentContextHierarchyBuilder | None,
) -> ContextualizedDocumentContent:
    """Build reusable hierarchy context and optional contextual chunk representations."""

    config = request.config
    parsed = request.parsed
    hierarchy, hierarchy_build_metadata = await _build_context_hierarchy(
        config=config,
        parsed=parsed,
        hierarchy_builder=hierarchy_builder,
        contextualizer=contextualizer,
    )
    contextualized_chunks, contextualization_metadata, contextualization_hierarchy = (
        await _contextualize_chunks(
            config=config,
            parsed=parsed,
            contextualizer=contextualizer,
            hierarchy=hierarchy,
        )
    )
    return ContextualizedDocumentContent(
        hierarchy=hierarchy or contextualization_hierarchy,
        hierarchy_build_metadata=hierarchy_build_metadata,
        contextualized_chunks=contextualized_chunks,
        contextualization_metadata=contextualization_metadata,
    )


async def _build_context_hierarchy(
    *,
    config: DocumentIngestionConfig,
    parsed: ParsedDocumentContent,
    hierarchy_builder: DocumentContextHierarchyBuilder | None,
    contextualizer: ChunkContextualizer | None,
) -> tuple[DocumentContextHierarchy | None, dict[str, object]]:
    required = config.contextualization_enabled or config.hierarchical_indexing_enabled
    if not required:
        return None, {"enabled": False, "status": "disabled"}
    if hierarchy_builder is None:
        if contextualizer is not None and config.contextualization_enabled:
            return None, {"enabled": True, "status": "deferred_to_contextualizer"}
        if config.hierarchical_indexing_fail_open:
            return None, {
                "enabled": True,
                "status": "failed_open",
                "error": "No document context hierarchy builder is configured.",
            }
        raise IngestionError(
            "Hierarchical indexing is enabled but no document context hierarchy builder is configured."
        )

    try:
        hierarchy = await hierarchy_builder.build(
            parsed.parsed_document,
            parsed.chunks,
            parsed.original_embeddings,
        )
    except Exception as exc:
        hierarchy_can_fail_open = (
            (not config.hierarchical_indexing_enabled or config.hierarchical_indexing_fail_open)
            and (not config.contextualization_enabled or config.contextualization_fail_open)
        )
        if not hierarchy_can_fail_open:
            logger.exception("Document context hierarchy generation failed; aborting ingestion.")
            raise
        logger.warning(
            "Document context hierarchy generation failed; continuing without hierarchy.",
            exc_info=True,
        )
        return None, {"enabled": True, "status": "failed_open", "error": str(exc)}

    return hierarchy, {
        "enabled": True,
        "status": "ready",
        "strategy": "semantic_cluster_hierarchy",
        "cluster_count": len(hierarchy.clusters),
    }


async def _contextualize_chunks(
    *,
    config: DocumentIngestionConfig,
    parsed: ParsedDocumentContent,
    contextualizer: ChunkContextualizer | None,
    hierarchy: DocumentContextHierarchy | None,
) -> tuple[list[ContextualizedChunk] | None, dict[str, object], DocumentContextHierarchy | None]:
    if not config.contextualization_enabled:
        logger.info(
            "Document contextualization is disabled; indexing original vectors only.",
            extra={"chunk_count": len(parsed.chunks)},
        )
        return None, {"enabled": False, "status": "disabled"}, hierarchy
    if contextualizer is None:
        raise IngestionError("Contextualization is enabled but no chunk contextualizer is configured.")

    representation_metadata = {
        "collection": config.vector_collection_name,
        "vector_name": config.contextual_vector_name,
    }
    logger.info(
        "Contextualizing chunks with the reusable document hierarchy before vector indexing.",
        extra={
            "document_title": parsed.parsed_document.title,
            "chunk_count": len(parsed.chunks),
            "contextual_vector_name": config.contextual_vector_name,
            "hierarchy_prebuilt": hierarchy is not None,
        },
    )
    try:
        contextualize = contextualizer.contextualize
        kwargs: dict[str, object] = {}
        if hierarchy is not None and "hierarchy" in inspect.signature(contextualize).parameters:
            kwargs["hierarchy"] = hierarchy
        result = await contextualize(
            parsed.parsed_document,
            parsed.chunks,
            parsed.original_embeddings,
            **kwargs,
        )
        contextualized_chunks = result.chunks
        if len(contextualized_chunks) != len(parsed.chunks):
            raise ValueError("Contextualizer must return exactly one representation per chunk.")
    except Exception as exc:
        if not config.contextualization_fail_open:
            logger.exception("Document contextualization failed; aborting ingestion.")
            raise
        logger.warning(
            "Document contextualization failed; continuing with original vectors only.",
            exc_info=True,
        )
        return None, {
            "enabled": True,
            "status": "failed_open",
            "error": str(exc),
            **representation_metadata,
        }, hierarchy

    logger.info(
        "Document contextualization completed.",
        extra={
            "document_title": parsed.parsed_document.title,
            "chunk_count": len(contextualized_chunks),
        },
    )
    resolved_hierarchy = hierarchy or result.hierarchy
    return contextualized_chunks, {
        "enabled": True,
        "status": "ready",
        "strategy": "semantic_cluster_hierarchy",
        "chunk_count": len(contextualized_chunks),
        "cluster_count": len(resolved_hierarchy.clusters),
        "document_summary": resolved_hierarchy.document_summary,
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "chunk_ordinals": list(cluster.chunk_ordinals),
                "summary": cluster.summary,
            }
            for cluster in resolved_hierarchy.clusters
        ],
        **representation_metadata,
    }, resolved_hierarchy
