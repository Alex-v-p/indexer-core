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
    structured_output_max_repair_attempts: int = Field(default=1, ge=0, le=1)
    temporal_query_timezone: str = "UTC"
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

    evidence_arbitration_fail_open: bool = True
    evidence_arbitration_relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence_arbitration_information_need_support_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence_arbitration_max_chars_per_evidence: int = Field(default=1_800, gt=0)
    evidence_arbitration_max_rationale_chars: int = Field(default=500, gt=0)

    primary_document_detection_enabled: bool = True
    primary_document_detection_min_score: float = Field(default=0.65, ge=0.0, le=1.0)
    primary_document_detection_min_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    primary_document_detection_replacement_margin: float = Field(default=0.12, ge=0.0, le=1.0)

    document_balancing_enabled: bool = True
    document_balancing_candidate_multiplier: int = Field(default=3, ge=1, le=20)
    document_balancing_max_candidates: int = Field(default=60, ge=1, le=500)
    document_balancing_primary_min_share: float = Field(default=0.60, gt=0.0, le=1.0)
    document_balancing_primary_max_share: float = Field(default=0.80, gt=0.0, le=1.0)
    document_balancing_secondary_max_share: float = Field(default=0.40, gt=0.0, le=1.0)
    document_balancing_unpreferred_max_share: float = Field(default=0.60, gt=0.0, le=1.0)

    retrieval_retry_max_retries: int = Field(default=2, ge=0, le=5)
    retrieval_retry_top_k_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    retrieval_retry_max_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_retry_expand_query: bool = True
    retrieval_retry_max_query_chars: int = Field(default=1_200, ge=100, le=8_000)
    retrieval_retry_max_total_attempts: int = Field(default=20, ge=1, le=200)
    retrieval_retry_max_reclassifications: int = Field(default=1, ge=0, le=3)
    retrieval_retry_max_accumulated_evidence: int = Field(default=40, ge=1, le=250)
    retrieval_retry_llm_rewrite_enabled: bool = True
    retrieval_retry_llm_rewrite_fail_open: bool = True
    retrieval_retry_llm_rewrite_max_attempts_in_prompt: int = Field(default=3, ge=1, le=10)
    retrieval_retry_llm_rewrite_max_evidence_per_attempt: int = Field(default=5, ge=1, le=20)
    retrieval_retry_llm_rewrite_max_chars_per_evidence: int = Field(default=700, ge=100, le=4_000)
    retrieval_retry_llm_rewrite_max_rationale_chars: int = Field(default=500, ge=100, le=2_000)
    retrieval_retry_llm_rewrite_max_missing_aspects: int = Field(default=5, ge=1, le=12)
    retrieval_retry_llm_rewrite_max_aspect_chars: int = Field(default=180, ge=40, le=500)

    database_url: str = Field(
        default="postgresql+asyncpg://indexer:indexer_password@localhost:5432/indexer",
        description="Async SQLAlchemy database URL.",
    )

    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_echo: bool = False

    bootstrap_db_max_attempts: int = 30
    bootstrap_db_retry_seconds: float = 2.0

    background_worker_poll_interval_seconds: float = Field(default=1, gt=0.0, le=60.0)
    background_worker_concurrency: int = Field(default=2, ge=1, le=32)
    background_worker_lock_timeout_seconds: int = Field(default=900, ge=30, le=86_400)
    background_worker_heartbeat_seconds: int = Field(default=30, ge=5, le=3_600)
    background_worker_retry_base_seconds: int = Field(default=15, ge=1, le=3_600)
    background_job_ingestion_max_attempts: int = Field(default=3, ge=1, le=10)
    background_job_maintenance_max_attempts: int = Field(default=2, ge=1, le=10)
    background_job_deletion_max_attempts: int = Field(default=3, ge=1, le=10)
    background_job_evaluation_max_attempts: int = Field(default=1, ge=1, le=10)
    background_job_query_max_attempts: int = Field(default=2, ge=1, le=10)
    background_job_query_priority: int = Field(default=25, ge=0, le=10_000)
    query_subject_scope_max_document_ids: int = Field(default=10_000, ge=1, le=100_000)
    query_subject_scope_max_project_lanes: int = Field(default=4, ge=1, le=8)
    query_subject_scope_policy_revision: str = Field(
        default="subject-scope-policy/1.0",
        min_length=1,
        max_length=128,
    )
    background_job_subject_classification_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
    )
    background_job_organization_classification_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
    )
    organization_classification_enabled: bool = True
    organization_classification_model_enabled: bool = True
    organization_classification_policy_version: str = "document-organization-policy/1.0"
    organization_classification_high_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    organization_classification_medium_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    organization_classification_confirmation_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    organization_classification_max_summary_chars: int = Field(default=2_000, ge=200, le=8_000)
    subject_classification_enabled: bool = False
    subject_classification_model_enabled: bool = True
    subject_classification_discovery_enabled: bool = True
    subject_classification_policy_version: str = "subject-decision-policy/3.7"
    subject_classification_high_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    subject_classification_medium_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    subject_classification_high_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    subject_classification_medium_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    subject_classification_minimum_suggestion_score: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )
    subject_classification_max_summary_chars: int = Field(default=2_000, ge=200, le=8_000)
    evaluation_dataset_dir: str = "./datasets/eval_sets"
    evaluation_report_dir: str = "./reports/evaluations"

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
    qdrant_hierarchy_vector_name: str = "hierarchy"
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

    hierarchical_indexing_enabled: bool = True
    hierarchical_indexing_fail_open: bool = False
    hierarchical_document_candidates: int = Field(default=8, ge=1, le=100)
    hierarchical_section_candidates: int = Field(default=24, ge=1, le=250)
    hierarchical_chunk_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    hierarchical_max_chunk_candidates: int = Field(default=80, ge=1, le=500)

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
    ollama_query_temperature: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    ollama_query_seed: int = 0
    ollama_query_max_output_tokens: int = Field(default=2_048, gt=0)
    ollama_structured_max_output_tokens: int = Field(default=4_096, gt=0)

    @model_validator(mode="after")
    def validate_cross_field_settings(self) -> "Settings":
        if (
            self.organization_classification_enabled
            and not self.organization_classification_model_enabled
        ):
            raise ValueError(
                "Organization classification cannot be enabled when its model provider is disabled.",
            )
        original = self.qdrant_original_vector_name.strip()
        contextual = self.qdrant_contextual_vector_name.strip()
        hierarchy = self.qdrant_hierarchy_vector_name.strip()
        vector_names = (original, contextual, hierarchy)
        if any(not name for name in vector_names):
            raise ValueError("Qdrant vector names must not be empty.")
        if len(set(vector_names)) != len(vector_names):
            raise ValueError("Original, contextual, and hierarchy Qdrant vector names must be different.")
        if (
            self.evidence_grading_information_need_support_threshold
            < self.evidence_grading_relevance_threshold
        ):
            raise ValueError(
                "Evidence information-need support threshold must be at least the relevance threshold.",
            )
        if (
            self.evidence_arbitration_information_need_support_threshold
            < self.evidence_arbitration_relevance_threshold
        ):
            raise ValueError(
                "Evidence arbitration support threshold must be at least the relevance threshold.",
            )
        if self.document_balancing_primary_min_share > self.document_balancing_primary_max_share:
            raise ValueError(
                "Document balancing primary minimum share cannot exceed the primary maximum share.",
            )
        if self.background_worker_heartbeat_seconds >= self.background_worker_lock_timeout_seconds:
            raise ValueError(
                "Background worker heartbeat interval must be shorter than the lock timeout.",
            )
        if (
            self.subject_classification_medium_threshold
            > self.subject_classification_high_threshold
        ):
            raise ValueError(
                "Subject classification medium threshold cannot exceed the high threshold.",
            )
        if (
            self.organization_classification_medium_threshold
            > self.organization_classification_high_threshold
        ):
            raise ValueError(
                "Organization classification medium threshold cannot exceed the high threshold.",
            )
        if self.retrieval_retry_max_total_attempts < self.retrieval_retry_max_retries + 1:
            raise ValueError(
                "The global retrieval-attempt budget must allow one information need to use its full retry budget.",
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
