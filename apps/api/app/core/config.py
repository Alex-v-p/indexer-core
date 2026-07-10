from functools import lru_cache
from typing import Literal

from pydantic import Field
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
        default="baseline_rag",
        description="Registered pipeline used when a query does not explicitly select one.",
    )

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
    qdrant_timeout_seconds: float = 30.0

    embedding_provider: Literal["ollama", "hashing"] = "ollama"
    embedding_vector_size: int = 768

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama HTTP API. Use http://ollama:11434 inside Docker Compose.",
    )
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout_seconds: float = 120.0

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
