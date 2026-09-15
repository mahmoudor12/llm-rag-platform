"""
Zentrale Konfiguration. Alle Werte kommen aus Umgebungsvariablen
oder der .env-Datei — kein Hardcoding.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    llm_provider: str = "ollama"          # "ollama" | "anthropic"
    llm_model: str = "qwen2.5:3b"
    ollama_host: str = "http://localhost:11434"
    anthropic_api_key: str | None = None

    # --- Vector Store ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Retrieval ---
    top_k: int = 5


settings = Settings()