"""API compatibility facade for provider dependency composition."""

from packages.indexer_bootstrap.composition.providers import (
    build_chunk_contextualizer,
    build_contextual_keyword_store,
    build_cross_encoder_reranker,
    build_document_context_hierarchy_builder,
    build_document_ingestion_config,
    build_document_object_store,
    build_embedding_provider,
    build_keyword_cache_invalidator,
    build_keyword_store,
    build_language_model,
    build_ollama_reranker,
    build_reranker,
    build_vector_store,
    require_structured_llm_provider,
)

__all__ = [
    "build_chunk_contextualizer",
    "build_contextual_keyword_store",
    "build_cross_encoder_reranker",
    "build_document_context_hierarchy_builder",
    "build_document_ingestion_config",
    "build_document_object_store",
    "build_embedding_provider",
    "build_keyword_cache_invalidator",
    "build_keyword_store",
    "build_language_model",
    "build_ollama_reranker",
    "build_reranker",
    "build_vector_store",
    "require_structured_llm_provider",
]
