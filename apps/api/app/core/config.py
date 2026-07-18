from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Keep infrastructure-specific values here so the rest of the application
    can depend on typed settings instead of reading environment variables
    directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Indexer Core API"
    environment: Literal["local", "development", "test", "staging", "production"] = "local"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    default_query_pipeline: str = Field(
        default="agentic_rag",
        description="Registered pipeline used when a query does not explicitly select one.",
    )

    query_classification_fail_open: bool = True
    query_classification_max_rationale_chars: int = 500
    retrieval_planning_low_confidence_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Factual classifications below this confidence use the reranking strategy.",
    )
    information_need_decomposition_fail_open: bool = True
    information_need_max_count: int = Field(default=6, ge=1, le=12)
    information_need_max_chars: int = Field(default=240, gt=0)
    information_need_decomposition_max_rationale_chars: int = Field(default=500, gt=0)

    evidence_grading_fail_open: bool = True
    evidence_grading_relevance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    evidence_grading_information_need_support_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    evidence_grading_max_chars_per_evidence: int = Field(default=2_000, gt=0)
    evidence_grading_max_rationale_chars: int = Field(default=500, gt=0)

    retrieval_retry_max_retries: int = Field(default=2, ge=0, le=5)
    retrieval_retry_top_k_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    retrieval_retry_max_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_retry_expand_query: bool = True
    retrieval_retry_max_query_chars: int = Field(default=1_200, ge=100, le=8_000)

    database_url: str = Field(
        default="postgresql+asyncpg://indexer:indexer_password@localhost:5432/indexer",
        description="Async SQLAlchemy database URL.",
    )

    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_echo: bool = False

    bootstrap_db_max_attempts: int = 30
    bootstrap_db_retry_seconds: float = 2.0

    document_storage_backend: Literal["minio", "local"] = Field(
        default="minio",
        description="Durable storage backend for uploaded source documents.",
    )
    document_storage_dir: str = Field(
        default="./storage/documents",
        description="Local storage directory when DOCUMENT_STORAGE_BACKEND=local.",
    )
    document_staging_dir: str = Field(
        default="/tmp/indexer-ingestion",
        description="Temporary staging directory used while parsing uploaded files.",
    )
    max_upload_size_mb: int = 25

    minio_endpoint: str = Field(
        default="localhost:9000",
        description="MinIO endpoint without a URL scheme. Use minio:9000 inside Docker Compose.",
    )
    minio_access_key: str = "indexer"
    minio_secret_key: str = "indexer_password"
    minio_bucket_name: str = "indexer-documents"
    minio_secure: bool = False
    minio_region: str | None = None
    minio_object_prefix: str = "documents"

    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Base URL for Qdrant. Use http://qdrant:6333 inside Docker Compose.",
    )
    qdrant_collection: str = "indexer_chunks"
    qdrant_original_vector_name: str = "original"
    qdrant_contextual_vector_name: str = "contextual"
    qdrant_timeout_seconds: float = 30.0

    keyword_scroll_batch_size: int = 256
    keyword_bm25_k1: float = 1.5
    keyword_bm25_b: float = 0.75
    keyword_cache_ttl_seconds: float = 30.0

    contextualization_enabled: bool = True
    contextualization_fail_open: bool = False
    contextualization_model: str = "llama3.2:3b"
    contextualization_neighbor_chunk_count: int = 2
    contextualization_max_neighbor_chars: int = 6_000
    contextualization_max_context_chars: int = 400
    contextualization_max_concurrency: int = 2
    contextualization_cluster_target_size: int = 8
    contextualization_max_clusters: int = 24
    contextualization_max_cluster_source_chars: int = 8_000
    contextualization_max_document_source_chars: int = 12_000
    contextualization_max_cluster_summary_chars: int = 600
    contextualization_max_document_summary_chars: int = 900

    hybrid_candidate_multiplier: int = 4
    hybrid_max_candidates: int = 100
    hybrid_rrf_k: int = 60
    hybrid_vector_weight: float = 1.0
    hybrid_keyword_weight: float = 1.0

    multi_query_variant_count: int = 3
    multi_query_include_original: bool = True
    multi_query_candidate_multiplier: int = 2
    multi_query_max_candidates_per_query: int = 20
    multi_query_rrf_k: int = 60
    multi_query_original_query_weight: float = 1.2
    multi_query_variant_query_weight: float = 1.0
    multi_query_max_variant_chars: int = 300
    multi_query_fail_open: bool = True

    rerank_candidate_multiplier: int = 4
    rerank_max_candidates: int = 40
    rerank_batch_size: int = 8
    rerank_max_chars_per_candidate: int = 4000
    ollama_rerank_max_attempts: int = 2
    ollama_rerank_fallback_to_original_rank: bool = True

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    cross_encoder_model_revision: str = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    cross_encoder_model_path: str = "/root/.cache/huggingface/indexer/cross-encoder"
    cross_encoder_local_files_only: bool = True
    cross_encoder_download_force: bool = False
    cross_encoder_batch_size: int = 16
    cross_encoder_max_length: int = 512
    cross_encoder_device: str = "cpu"

    embedding_provider: Literal["ollama", "hashing"] = "ollama"
    embedding_vector_size: int = 768

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama HTTP API. Use http://ollama:11434 inside Docker Compose.",
    )
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_rerank_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 120.0

    @model_validator(mode="after")
    def validate_cross_field_settings(self) -> "Settings":
        original = self.qdrant_original_vector_name.strip()
        contextual = self.qdrant_contextual_vector_name.strip()
        if not original or not contextual:
            raise ValueError("Qdrant vector names must not be empty.")
        if original == contextual:
            raise ValueError("Original and contextual Qdrant vector names must be different.")
        if (
            self.evidence_grading_information_need_support_threshold
            < self.evidence_grading_relevance_threshold
        ):
            raise ValueError(
                "Evidence information-need support threshold must be at least the relevance threshold.",
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
