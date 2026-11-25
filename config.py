from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/aidb"

    # Local Ollama & LLM Settings
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_url: str = "http://127.0.0.1:11434/api/chat"
    llm_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"

    # Optional OpenAI-compatible endpoint settings (used by /ask/)
    llm_api_url: str = "http://127.0.0.1:11434/v1/chat/completions"
    llm_api_key: str = "ollama"

    # Chunking Parameters
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Backend URL (for Streamlit frontend)
    api_base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()