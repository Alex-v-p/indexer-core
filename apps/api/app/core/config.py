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

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
