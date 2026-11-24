import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from app.core.config import settings
from app.core.logging import logger
from app.services.embedding_factory import EmbeddingFactory


class VectorStoreService:
    """
    Manages vector database lifecycle, document indexing, similarity search,
    and automatic disk persistence.
    """

    def __init__(self):
        self.embedding_model = EmbeddingFactory.get_embedding_model()
        self.persist_dir = Path(settings.PERSIST_DIRECTORY)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.persist_dir / "in_memory_index.json"
        
        self.vector_store: VectorStore = None
        self._all_documents: List[Document] = []
        
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration."""
        db_type = settings.VECTOR_DB_TYPE.lower()
        logger.info("Initializing vector store: %s", db_type)

        if db_type == "chroma":
            try:
                from langchain_community.vectorstores import Chroma
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_dir / "chroma"),
                    embedding_function=self.embedding_model
                )
                logger.info("Chroma vector store initialized.")
                return
            except Exception as e:
                logger.warning("ChromaDB initialization failed (%s). Falling back to InMemoryVectorStore.", e)

        elif db_type == "faiss":
            try:
                from langchain_community.vectorstores import FAISS
                faiss_path = self.persist_dir / "faiss_index"
                if (faiss_path / "index.faiss").exists():
                    self.vector_store = FAISS.load_local(
                        str(faiss_path),
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                else:
                    self.vector_store = None  # Will be built on first add
                logger.info("FAISS vector store initialized.")
                return
            except Exception as e:
                logger.warning("FAISS initialization failed (%s). Falling back to InMemoryVectorStore.", e)

        # Default: InMemoryVectorStore with JSON disk backup
        self.vector_store = InMemoryVectorStore(embedding=self.embedding_model)
        self._load_from_disk()

    def add_documents(self, documents: List[Document]) -> int:
        """Add chunks to the vector store and persist them."""
        if not documents:
            return 0

        self.vector_store.add_documents(documents)
        self._all_documents.extend(documents)
        self._save_to_disk()

        logger.info(
            "Added %d chunks to vector store. Total chunks: %d",
            len(documents), len(self._all_documents)
        )
        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Execute similarity search and filter by score threshold.
        Returns list of (Document, score) tuples sorted by descending similarity.
        """
        top_k = k or settings.TOP_K
        min_threshold = threshold if threshold is not None else settings.SIMILARITY_SCORE_THRESHOLD

        if not self._all_documents:
            logger.debug("Vector store is empty, returning empty search results.")
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            logger.error("Error during similarity search: %s", e)
            return []

        # Filter by threshold if requested
        filtered = []
        for doc, score in results:
            if min_threshold is None or score >= min_threshold:
                filtered.append((doc, float(score)))

        logger.debug(
            "Query '%s' returned %d chunks (threshold=%.2f, raw=%d)",
            query, len(filtered), min_threshold, len(results)
        )
        return filtered

    def get_stats(self) -> dict:
        """Return summary statistics about indexed content."""
        sources = set()
        for doc in self._all_documents:
            src = doc.metadata.get("source")
            if src:
                sources.add(src)

        return {
            "vector_db_type": settings.VECTOR_DB_TYPE,
            "total_chunks": len(self._all_documents),
            "unique_sources": sorted(list(sources)),
            "storage_path": str(self.persist_dir.resolve())
        }

    def clear(self) -> None:
        """Clear all indexed documents from memory and disk."""
        self._all_documents.clear()
        self.vector_store = InMemoryVectorStore(embedding=self.embedding_model)
        if self.index_file.exists():
            try:
                self.index_file.unlink()
                logger.info("Deleted persisted index file: %s", self.index_file)
            except Exception as e:
                logger.error("Failed to delete index file: %s", e)
        logger.info("Vector store reset complete.")

    def _save_to_disk(self):
        """Serialize documents to JSON on disk."""
        try:
            data = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in self._all_documents
            ]
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("Persisted %d chunks to %s", len(data), self.index_file)
        except Exception as e:
            logger.error("Failed to persist vector store to disk: %s", e)

    def _load_from_disk(self):
        """Reload documents from JSON disk backup on startup."""
        if not self.index_file.exists():
            return

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            docs = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in data
            ]
            if docs:
                self.vector_store.add_documents(docs)
                self._all_documents = docs
                logger.info("Loaded %d persisted chunks from %s", len(docs), self.index_file)
        except Exception as e:
            logger.error("Failed to load persisted vector store from disk: %s", e)


# Global singleton instance
vector_store_service = VectorStoreService()
