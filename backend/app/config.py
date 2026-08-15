"""
config.py
Single source of truth for all environment/config values. Every other
module imports `settings` from here instead of calling os.environ
directly — this means (a) all config is documented in one place,
(b) missing required values fail loudly at startup instead of silently
at request time, and (c) tests can override settings easily.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- LLM (Groq) ---
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-large-en-v1.5"

    # --- Storage ---
    chroma_persist_dir: str = "./vectorstore"
    raw_filings_dir: str = "./data/raw_filings"

    # --- CORS ---
    allowed_origins: str = "http://localhost:5173"

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]


settings = Settings()
