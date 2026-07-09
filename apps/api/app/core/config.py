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

    database_url: str = Field(
        default="postgresql+asyncpg://indexer:indexer_password@localhost:5432/indexer",
        description="Async SQLAlchemy database URL.",
    )

    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_echo: bool = False

    bootstrap_db_max_attempts: int = 30
    bootstrap_db_retry_seconds: float = 2.0

    document_storage_dir: str = Field(
        default="./storage/documents",
        description="Local directory where uploaded source documents are stored.",
    )
    max_upload_size_mb: int = 25

    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Base URL for Qdrant. Use http://qdrant:6333 inside Docker Compose.",
    )
    qdrant_collection: str = "indexer_chunks"
    qdrant_timeout_seconds: float = 30.0

    embedding_provider: Literal["hashing"] = "hashing"
    embedding_vector_size: int = 384

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama HTTP API. Use http://ollama:11434 inside Docker Compose.",
    )
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: float = 120.0

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
