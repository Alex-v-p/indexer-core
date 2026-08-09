from __future__ import annotations

from packages.indexer_bootstrap.config import Settings
from packages.indexer_application.dto import DocumentIngestionConfig
from packages.indexer_application.ports import DocumentObjectStore
from packages.indexer_infrastructure.minio import MinioDocumentObjectStore
from packages.indexer_infrastructure.object_storage import LocalDocumentObjectStore
from packages.indexer_infrastructure.ollama import OllamaLLMProvider
from packages.rag_core.ingestion import (
    ChunkContextualizer,
    ContextHierarchyConfig,
    ContextualizationConfig,
    LLMChunkContextualizer,
    LLMDocumentContextHierarchyBuilder,
)
from packages.rag_core.subjects import SubjectClassificationPolicy


def build_document_context_hierarchy_builder(
    settings: Settings,
) -> LLMDocumentContextHierarchyBuilder | None:
    if not settings.contextualization_enabled and not settings.hierarchical_indexing_enabled:
        return None
    hierarchy_llm = OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.contextualization_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    return LLMDocumentContextHierarchyBuilder(
        llm_provider=hierarchy_llm,
        config=ContextHierarchyConfig(
            target_cluster_size=settings.contextualization_cluster_target_size,
            max_clusters=settings.contextualization_max_clusters,
            max_cluster_source_chars=settings.contextualization_max_cluster_source_chars,
            max_document_source_chars=settings.contextualization_max_document_source_chars,
            max_cluster_summary_chars=settings.contextualization_max_cluster_summary_chars,
            max_document_summary_chars=settings.contextualization_max_document_summary_chars,
            max_concurrency=settings.contextualization_max_concurrency,
        ),
    )


def build_chunk_contextualizer(
    settings: Settings,
    *,
    hierarchy_builder: LLMDocumentContextHierarchyBuilder | None = None,
) -> ChunkContextualizer | None:
    if not settings.contextualization_enabled:
        return None

    contextualization_llm = OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.contextualization_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    resolved_hierarchy_builder = hierarchy_builder or build_document_context_hierarchy_builder(settings)
    if resolved_hierarchy_builder is None:
        raise ValueError("Contextualization requires a document context hierarchy builder.")
    return LLMChunkContextualizer(
        llm_provider=contextualization_llm,
        hierarchy_builder=resolved_hierarchy_builder,
        config=ContextualizationConfig(
            neighbor_chunk_count=settings.contextualization_neighbor_chunk_count,
            max_neighbor_chars=settings.contextualization_max_neighbor_chars,
            max_context_chars=settings.contextualization_max_context_chars,
            max_concurrency=settings.contextualization_max_concurrency,
        ),
    )


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
        original_vector_name=settings.qdrant_original_vector_name,
        contextual_vector_name=settings.qdrant_contextual_vector_name,
        hierarchy_vector_name=settings.qdrant_hierarchy_vector_name,
        contextualization_enabled=settings.contextualization_enabled,
        contextualization_fail_open=settings.contextualization_fail_open,
        hierarchical_indexing_enabled=settings.hierarchical_indexing_enabled,
        hierarchical_indexing_fail_open=settings.hierarchical_indexing_fail_open,
    )


def build_subject_classification_policy(
    settings: Settings,
) -> SubjectClassificationPolicy:
    return SubjectClassificationPolicy(
        high_threshold=settings.subject_classification_high_threshold,
        medium_threshold=settings.subject_classification_medium_threshold,
        high_margin=settings.subject_classification_high_margin,
        medium_margin=settings.subject_classification_medium_margin,
        minimum_suggestion_score=(
            settings.subject_classification_minimum_suggestion_score
        ),
        policy_version=settings.subject_classification_policy_version,
    )


def build_document_organization_policy(settings: Settings):
    from packages.rag_core.document_organization import DocumentOrganizationPolicy

    return DocumentOrganizationPolicy(
        high_threshold=settings.organization_classification_high_threshold,
        medium_threshold=settings.organization_classification_medium_threshold,
        confirmation_margin=settings.organization_classification_confirmation_margin,
        policy_version=settings.organization_classification_policy_version,
    )
