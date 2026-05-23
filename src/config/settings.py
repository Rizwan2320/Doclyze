from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file manually BEFORE creating Settings (this fixes the issue)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=True)
    print(f"✅ Loaded .env from: {ENV_FILE}")
else:
    print(f"⚠️  .env file not found at: {ENV_FILE}")


class Settings(BaseSettings):
    """All configuration settings for the Doclyze application."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        # We removed env_file here because we load it manually
    )

    # ── Core secrets ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: SecretStr = Field(
        default=...,
        description="API key for Groq inference",
    )

    # ── Embedding model ──────────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    EMBEDDING_DIMENSION: int = Field(default=384, frozen=True)

    # ── Vector store (Chroma) ────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = Field(
        default=str(PROJECT_ROOT / "chroma_db"),
    )

    # ── Ingestion / Chunking defaults ────────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=1000, ge=100, le=2000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=500)

    # ── Logging & observability ──────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_API_KEY: SecretStr | None = Field(default=None)

    # ── Groq / LLM defaults ──────────────────────────────────────────────────────
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    GROQ_MAX_TOKENS: int = Field(default=4096, ge=128)


# Singleton
settings = Settings()