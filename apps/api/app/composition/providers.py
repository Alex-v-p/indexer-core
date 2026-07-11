from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings
from packages.indexer_application.dto import DocumentIngestionConfig
from packages.indexer_application.ports import CacheInvalidator, DocumentObjectStore
from packages.indexer_infrastructure.bm25 import BM25KeywordStore
from packages.indexer_infrastructure.cross_encoder import CrossEncoderReranker
from packages.indexer_infrastructure.embeddings import HashingEmbeddingProvider
from packages.indexer_infrastructure.minio import MinioDocumentObjectStore
from packages.indexer_infrastructure.object_storage import LocalDocumentObjectStore
from packages.indexer_infrastructure.ollama import OllamaEmbeddingProvider, OllamaLLMProvider, OllamaReranker
from packages.indexer_infrastructure.qdrant import QdrantKeywordCorpusSource, QdrantVectorStore
from packages.rag_core.ports import EmbeddingProvider, LLMProvider, VectorStore
from packages.rag_core.retrieval.rerankers import Reranker


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(vector_size=settings.embedding_vector_size)
    return OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        vector_size=settings.embedding_vector_size,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


def build_language_model(settings: Settings) -> LLMProvider:
    return OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


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
        model_name=settings.cross_encoder_model,
        batch_size=settings.cross_encoder_batch_size,
        max_length=settings.cross_encoder_max_length,
        device=settings.cross_encoder_device,
        cache_folder=settings.cross_encoder_cache_dir,
    )


def build_reranker(settings: Settings) -> Reranker:
    """Backward-compatible alias for the existing Ollama reranker."""

    return build_ollama_reranker(settings)


def build_vector_store(settings: Settings) -> VectorStore:
    return QdrantVectorStore(
        base_url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_vector_size,
        timeout_seconds=settings.qdrant_timeout_seconds,
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
    )


@lru_cache(maxsize=8)
def _build_qdrant_bm25_keyword_store(
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_timeout_seconds: float,
    scroll_batch_size: int,
    k1: float,
    b: float,
    cache_ttl_seconds: float,
) -> BM25KeywordStore:
    source = QdrantKeywordCorpusSource(
        base_url=qdrant_url,
        collection_name=qdrant_collection,
        timeout_seconds=qdrant_timeout_seconds,
        scroll_batch_size=scroll_batch_size,
    )
    return BM25KeywordStore(
        corpus_source=source,
        k1=k1,
        b=b,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def build_keyword_cache_invalidator(settings: Settings) -> CacheInvalidator:
    return build_keyword_store(settings)


def build_document_object_store(settings: Settings) -> DocumentObjectStore:
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024 if settings.max_upload_size_mb > 0 else None
    if settings.document_storage_backend == "local":
        return LocalDocumentObjectStore(base_dir=settings.document_storage_dir, max_size_bytes=max_size_bytes)
    return MinioDocumentObjectStore(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name=settings.minio_bucket_name,
        secure=settings.minio_secure,
        region=settings.minio_region or None,
        object_prefix=settings.minio_object_prefix,
        staging_dir=settings.document_staging_dir,
        max_size_bytes=max_size_bytes,
    )


def build_document_ingestion_config(settings: Settings) -> DocumentIngestionConfig:
    return DocumentIngestionConfig(
        chunk_max_chars=settings.chunk_max_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        vector_collection_name=settings.qdrant_collection,
    )
