import os
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application Info
    APP_NAME: str = "Enterprise RAG API"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # LLM Provider Configuration
    # Options: "openai", "google", "ollama", "groq", "mock"
    LLM_PROVIDER: Literal["openai", "google", "ollama", "groq", "mock"] = "groq"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048

    # Embedding Provider Configuration
    # Options: "openai", "google", "ollama", "huggingface", "tfidf"
    EMBEDDING_PROVIDER: Literal["openai", "google", "ollama", "huggingface", "tfidf"] = "tfidf"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # API Keys & Endpoints
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Vector Database Configuration
    # Options: "in_memory", "chroma", "faiss"
    VECTOR_DB_TYPE: Literal["in_memory", "chroma", "faiss"] = "in_memory"
    VECTOR_STORE_DIR: str = os.path.join("data", "vector_store_data")
    PERSIST_DIRECTORY: str = os.path.join("data", "vector_store_data")

    # Document Chunking Settings
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100

    # Retrieval Settings
    TOP_K: int = 4
    SIMILARITY_SCORE_THRESHOLD: float = 0.05

    # System Prompt Constraints
    SYSTEM_PROMPT: str = (
        "You are an expert AI assistant answering questions strictly based on provided context documents.\n"
        "Guidelines:\n"
        "1. Answer truthfully and accurately using ONLY the information in the context.\n"
        "2. If the context does not contain enough information to answer, state clearly: "
        "'Based on the provided documents, I do not have enough information to answer this question.'\n"
        "3. Do not make up facts, URLs, or citations.\n"
        "4. Always reference relevant document sources when answering."
    )


settings = Settings()
