"""Application configuration settings."""

from typing import Any, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Project info
    PROJECT_NAME: str = "Chives Agentic Framework"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # Server config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # CORS
    ALLOWED_HOSTS: list[str] = ["*"]

    # LLM Configuration - REQUIRED environment variables
    OPENAPI_LLM_URL: str
    OPENAPI_LLM_API_KEY: str
    OPENAPI_LLM_MODEL: str

    # ChromaDB Configuration for RAG
    CHROMA_COLLECTION_NAME: str = "documents"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_HOST: Optional[str] = None
    CHROMA_PORT: Optional[int] = None
    # Note: These variables match the chives-ingestion submodule naming convention


settings = Settings()
