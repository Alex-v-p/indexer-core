"""API compatibility facade for document dependency composition."""

from packages.indexer_bootstrap.composition.documents import (
    build_chunk_contextualizer,
    build_document_context_hierarchy_builder,
    build_document_ingestion_config,
    build_document_object_store,
)

__all__ = [
    "build_chunk_contextualizer",
    "build_document_context_hierarchy_builder",
    "build_document_ingestion_config",
    "build_document_object_store",
]
