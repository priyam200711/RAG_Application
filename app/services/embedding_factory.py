import math
import re
from typing import List
from langchain_core.embeddings import Embeddings
from app.core.config import settings
from app.core.logging import logger


class SemanticHashEmbeddings(Embeddings):
    """
    Deterministic, high-performance 128-dimensional embedding model
    using token n-grams and hashing trick with L2 normalization.
    Ensures zero external dependency and zero-cost local execution.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _hash_token(self, token: str) -> int:
        h = 2166136261
        for char in token:
            h = (h ^ ord(char)) * 16777619
        return (h & 0xFFFFFFFF) % self.dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return vec

        # Unigrams and bigrams
        for i, tok in enumerate(tokens):
            vec[self._hash_token(tok)] += 1.0
            if i < len(tokens) - 1:
                bigram = f"{tok}_{tokens[i + 1]}"
                vec[self._hash_token(bigram)] += 1.5

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class EmbeddingFactory:
    """Factory creating Embeddings instances based on configured provider."""

    _instance: Embeddings = None

    @classmethod
    def get_embedding_model(cls) -> Embeddings:
        if cls._instance is not None:
            return cls._instance

        provider = settings.EMBEDDING_PROVIDER.lower()
        logger.info("Initializing embedding model for provider: %s", provider)

        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                cls._instance = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    api_key=settings.OPENAI_API_KEY
                )
            except Exception as e:
                logger.warning("Failed to initialize OpenAIEmbeddings (%s). Falling back to SemanticHashEmbeddings.", e)
                cls._instance = SemanticHashEmbeddings()

        elif provider == "google":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                cls._instance = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=settings.GOOGLE_API_KEY
                )
            except Exception as e:
                logger.warning("Failed to initialize Google embeddings (%s). Falling back to SemanticHashEmbeddings.", e)
                cls._instance = SemanticHashEmbeddings()

        elif provider == "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings
                cls._instance = OllamaEmbeddings(
                    model=settings.EMBEDDING_MODEL or "nomic-embed-text",
                    base_url=settings.OLLAMA_BASE_URL
                )
            except Exception as e:
                logger.warning("Failed to initialize OllamaEmbeddings (%s). Falling back to SemanticHashEmbeddings.", e)
                cls._instance = SemanticHashEmbeddings()

        elif provider == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                cls._instance = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
            except Exception as e:
                logger.warning("Failed to initialize HuggingFaceEmbeddings (%s). Falling back to SemanticHashEmbeddings.", e)
                cls._instance = SemanticHashEmbeddings()

        else:
            cls._instance = SemanticHashEmbeddings()

        return cls._instance

    @classmethod
    def reset(cls):
        """Reset cached instance (useful for unit tests and configuration changes)."""
        cls._instance = None
