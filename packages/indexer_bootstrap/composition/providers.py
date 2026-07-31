from __future__ import annotations

from functools import lru_cache

from packages.indexer_bootstrap.config import Settings
from packages.indexer_bootstrap.composition.documents import (
    build_chunk_contextualizer,
    build_document_context_hierarchy_builder,
    build_document_ingestion_config,
    build_document_object_store,
)
from packages.indexer_application.ports import CacheInvalidator
from packages.indexer_infrastructure.bm25 import BM25KeywordStore
from packages.indexer_infrastructure.cross_encoder import CrossEncoderReranker
from packages.indexer_infrastructure.embeddings import HashingEmbeddingProvider
from packages.indexer_infrastructure.ollama import OllamaEmbeddingProvider, OllamaLLMProvider, OllamaReranker
from packages.indexer_infrastructure.qdrant import QdrantKeywordCorpusSource, QdrantVectorStore
from packages.rag_core.ports import (
    EmbeddingProvider,
    LLMProvider,
    StructuredLLMProvider,
    VectorStore,
)
from packages.rag_core.retrieval.rerankers import Reranker


class _CompositeCacheInvalidator:
    def __init__(self, *invalidators: CacheInvalidator) -> None:
        self._invalidators = invalidators

    def invalidate(self) -> None:
        for invalidator in self._invalidators:
            invalidator.invalidate()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(vector_size=settings.embedding_vector_size)
    return OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        vector_size=settings.embedding_vector_size,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


def require_structured_llm_provider(provider: LLMProvider) -> StructuredLLMProvider:
    if not isinstance(provider, StructuredLLMProvider):
        raise TypeError(
            "The configured query language model does not support structured generation.",
        )
    return provider


def build_language_model(settings: Settings) -> StructuredLLMProvider:
    provider = OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        temperature=settings.ollama_query_temperature,
        seed=settings.ollama_query_seed,
        max_output_tokens=settings.ollama_query_max_output_tokens,
        structured_max_output_tokens=settings.ollama_structured_max_output_tokens,
    )
    return require_structured_llm_provider(provider)


def build_ollama_reranker(settings: Settings) -> Reranker:
    return OllamaReranker(
        base_url=settings.ollama_base_url,
        model=settings.ollama_rerank_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        batch_size=settings.rerank_batch_size,
        max_chars_per_candidate=settings.rerank_max_chars_per_candidate,
        max_attempts=settings.ollama_rerank_max_attempts,
        fallback_to_original_rank=settings.ollama_rerank_fallback_to_original_rank,
    )


def build_cross_encoder_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(
        model_name=settings.cross_encoder_model_path,
        model_identifier=settings.cross_encoder_model,
        batch_size=settings.cross_encoder_batch_size,
        max_length=settings.cross_encoder_max_length,
        device=settings.cross_encoder_device,
        local_files_only=settings.cross_encoder_local_files_only,
    )


def build_reranker(settings: Settings) -> Reranker:
    """Backward-compatible alias for the existing Ollama reranker."""

    return build_ollama_reranker(settings)


def build_vector_store(settings: Settings) -> VectorStore:
    return _build_vector_store(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.embedding_vector_size,
        settings.qdrant_original_vector_name,
        settings.qdrant_contextual_vector_name,
        settings.qdrant_hierarchy_vector_name,
        settings.qdrant_timeout_seconds,
    )


@lru_cache(maxsize=8)
def _build_vector_store(
    qdrant_url: str,
    collection_name: str,
    vector_size: int,
    original_vector_name: str,
    contextual_vector_name: str,
    hierarchy_vector_name: str,
    timeout_seconds: float,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        base_url=qdrant_url,
        collection_name=collection_name,
        vector_size=vector_size,
        vector_names=(original_vector_name, contextual_vector_name, hierarchy_vector_name),
        timeout_seconds=timeout_seconds,
    )


def build_keyword_store(settings: Settings) -> BM25KeywordStore:
    return _build_qdrant_bm25_keyword_store(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.qdrant_timeout_seconds,
        settings.keyword_scroll_batch_size,
        settings.keyword_bm25_k1,
        settings.keyword_bm25_b,
        settings.keyword_cache_ttl_seconds,
        "text",
        True,
    )


def build_contextual_keyword_store(settings: Settings) -> BM25KeywordStore:
    return _build_qdrant_bm25_keyword_store(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.qdrant_timeout_seconds,
        settings.keyword_scroll_batch_size,
        settings.keyword_bm25_k1,
        settings.keyword_bm25_b,
        settings.keyword_cache_ttl_seconds,
        "contextualized_text",
        False,
    )


@lru_cache(maxsize=16)
def _build_qdrant_bm25_keyword_store(
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_timeout_seconds: float,
    scroll_batch_size: int,
    k1: float,
    b: float,
    cache_ttl_seconds: float,
    search_text_field: str,
    fallback_to_evidence_text: bool,
) -> BM25KeywordStore:
    source = QdrantKeywordCorpusSource(
        base_url=qdrant_url,
        collection_name=qdrant_collection,
        timeout_seconds=qdrant_timeout_seconds,
        scroll_batch_size=scroll_batch_size,
        search_text_field=search_text_field,
        evidence_text_field="text",
        fallback_to_evidence_text=fallback_to_evidence_text,
    )
    return BM25KeywordStore(
        corpus_source=source,
        k1=k1,
        b=b,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def build_keyword_cache_invalidator(settings: Settings) -> CacheInvalidator:
    return _CompositeCacheInvalidator(
        build_keyword_store(settings),
        build_contextual_keyword_store(settings),
    )


