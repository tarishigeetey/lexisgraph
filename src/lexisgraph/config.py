"""Central, typed configuration. All values come from environment / .env.

Nothing else in the codebase hardcodes a host, port, or model name — they
read from here. That single source of truth is what makes the system
deployable to different environments without code changes.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Qdrant (vector store) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "legal_clauses"

    # --- Embedding model ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Chunking ---
    chunk_size: int = 800       # characters per chunk
    chunk_overlap: int = 150    # overlap between consecutive chunks

    # --- LLM (Ollama, local) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # --- Data ---
    cuad_dir: str = "data/cuad"  # override in .env with your real path


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so we build Settings once per process."""
    return Settings()
