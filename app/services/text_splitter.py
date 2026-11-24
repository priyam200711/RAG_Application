from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logging import logger


class DocumentChunker:
    """Splits documents into overlapping chunks using RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(
        self,
        text: str,
        source: str,
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """Split a raw text string into a list of Document chunks with rich metadata."""
        base_meta = metadata.copy() if metadata else {}
        base_meta["source"] = source

        raw_chunks = self.splitter.split_text(text)
        total_chunks = len(raw_chunks)
        
        documents = []
        for idx, chunk in enumerate(raw_chunks):
            chunk_meta = {
                **base_meta,
                "chunk_id": idx,
                "total_chunks": total_chunks,
                "chunk_size_chars": len(chunk),
            }
            documents.append(Document(page_content=chunk, metadata=chunk_meta))

        logger.debug(
            "Split text from '%s' (%d chars) into %d chunks (size=%d, overlap=%d)",
            source, len(text), total_chunks, self.chunk_size, self.chunk_overlap
        )
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split existing LangChain Document objects."""
        chunked_docs = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            chunks = self.split_text(
                text=doc.page_content,
                source=source,
                metadata=doc.metadata
            )
            chunked_docs.extend(chunks)
        return chunked_docs
